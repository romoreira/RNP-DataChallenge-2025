# svdd_mlab_oneclass_gpu_calibrated_final.py
# Deep-SVDD one-class (FIXED-R + calibração) para mudança de rota (M-Lab)
# - Featurização rápida (cache/chunks opcionais)
# - Warm-up curto
# - R fixo por quantil (1-NU) -> estável
# - Calibração do limiar NA VALIDAÇÃO usando o MODELO FINAL
# - Gera submission.csv

import os, time, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# =======================
# Configurações principais
# =======================
SEED = 42
VAL_SIZE = 0.20

NU = 0.05                 # fração de outliers tolerada entre normais
WARMUP_EPOCHS = 1         # warm-up curto
EPOCHS = 10                # sem refino longo (evita colapso)
LR = 1e-3
WD = 1e-5
BATCH_TRAIN = 2048
BATCH_EVAL = 4096

USE_FIXED_R = True        # usamos FIXED-R + calibração final

CACHE_DIR = "./_cache_feats"
USE_CACHE = True
CHUNKSIZE = None          # ex.: 1_000_000 para processar CSV em partes
CSV_ENGINE = "pyarrow"    # "pyarrow" (rápido) ou "c" (fallback)
TRAIN_CSV = "dataset/train.csv"
TEST_CSV  = "dataset/test.csv"
NROWS = None              # None = tudo; ou defina um número p/ teste

# =======================
# Utilidades
# =======================
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_csv_fast(path, nrows=None, use_engine="pyarrow"):
    t0 = time.time()
    kw = dict(nrows=nrows)
    if use_engine == "pyarrow":
        try:
            df = pd.read_csv(path, engine="pyarrow", **kw)
            print(f"[INFO] read_csv (pyarrow) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
            return df
        except Exception as e:
            print(f"[WARN] pyarrow falhou ({e}). Caindo para engine='c'.")
    df = pd.read_csv(path, engine="c", **kw)
    print(f"[INFO] read_csv (c) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

# =======================
# Featurização RÁPIDA
# =======================
def fast_parse_array(s: str):
    if not isinstance(s, str) or len(s) < 2:
        return np.array([], dtype=np.float32)
    return np.fromstring(s[1:-1], sep=',', dtype=np.float32)

def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)

    means   = np.fromiter((a.mean()               if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds    = np.fromiter((a.std(ddof=0)          if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    medians = np.fromiter((np.median(a)           if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p90s    = np.fromiter((np.quantile(a, 0.9)    if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    mins    = np.fromiter((a.min()                if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    maxs    = np.fromiter((a.max()                if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    lens    = np.fromiter((a.size                 for a in rtts),                      dtype=np.float32, count=n)

    date_index_series = df["date_index"] if "date_index" in df.columns else pd.Series(0, index=df.index)

    out = pd.DataFrame({
        "rtt_mean": means, "rtt_std": stds, "rtt_median": medians,
        "rtt_p90": p90s, "rtt_min": mins, "rtt_max": maxs, "rtt_len": lens,
        "tr_attempts": df["tr_attempts"].astype(np.float32),
        "total_probes_sent": df["total_probes_sent"].astype(np.float32),
        "total_replies_last_hop": df["total_replies_last_hop"].astype(np.float32),
        "seconds_since_start": df["seconds_since_start"].astype(np.float32),
        "date_index": date_index_series.astype(np.float32),
    })
    eps = 1e-9
    out["success_rate"] = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"] = 1.0 - out["success_rate"]
    out["replies_per_attempt"] = out["total_replies_last_hop"] / (out["tr_attempts"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print(f"[INFO] featurize_fast: {out.shape} | {time.time()-t0:.1f}s")
    return out

def featurize_csv_in_chunks(path, chunksize=1_000_000, nrows=None):
    t0 = time.time()
    dfs, read_rows = [], 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize)):
        if nrows is not None and read_rows >= nrows: break
        if nrows is not None:
            take = max(0, min(nrows - read_rows, len(chunk)))
            chunk = chunk.iloc[:take]
        print(f"[INFO] Chunk {i} lido: {chunk.shape}")
        dfs.append(featurize_fast(chunk))
        read_rows += len(chunk)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"[INFO] featurize_csv_in_chunks: {out.shape} | {time.time()-t0:.1f}s")
    return out

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if USE_CACHE and os.path.exists(cache_path):
        try:
            t0 = time.time()
            df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) == nrows):
                print(f"[INFO] Carregado do cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
                return df
            else:
                print("[INFO] Cache existe mas nrows difere; recalculando features.")
        except Exception as e:
            print(f"[WARN] Falha ao ler cache ({e}). Recriando.")
    base = read_csv_fast(csv_path, nrows=nrows, use_engine=CSV_ENGINE)
    feats = featurize_fast(base) if CHUNKSIZE is None else featurize_csv_in_chunks(csv_path, CHUNKSIZE, nrows)
    if USE_CACHE:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats.to_parquet(cache_path, index=False)
        print(f"[INFO] Cache salvo em: {cache_path}")
    return feats

# =======================
# Dataset simples
# =======================
class NPDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None: return (self.X[i],)
        return self.X[i], self.y[i]

# =======================
# Deep-SVDD (one-class puro)
# =======================
class DeepSVDDNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)  # z(x)
        )
    def forward(self, x): return self.net(x)



@torch.no_grad()
def init_center_c(net, loader, device):
    net.eval()
    c = torch.zeros(32, device=device)
    n = 0
    for (xb,) in loader:
        xb = xb.to(device)
        z = net(xb)
        c += z.sum(0)
        n += z.size(0)
    c /= max(n, 1)
    c[torch.isclose(c, torch.zeros_like(c), atol=1e-6)] = 1e-6
    return c

@torch.no_grad()
def compute_distances_on_loader(net, loader, c, device):
    net.eval()
    dists = []
    for (xb,) in loader:
        xb = xb.to(device)
        z = net(xb)
        d = torch.sum((z - c)**2, dim=1)
        dists.append(d.detach().cpu())
    return torch.cat(dists, dim=0) if dists else torch.tensor([], dtype=torch.float32)

def warmup_minimize_mean_distance(net, dl_norm, c, epochs=1, lr=1e-3, wd=1e-5, device="cuda"):
    net.to(device).train()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, epochs+1):
        t0, ep_loss = time.time(), 0.0
        for (xb,) in dl_norm:
            xb = xb.to(device)
            z = net(xb)
            dist = torch.sum((z - c)**2, dim=1)
            loss = dist.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        print(f"[INFO] Warmup {ep:02d}/{epochs} | mean_dist={ep_loss/len(dl_norm):.6f} | {time.time()-t0:.1f}s")

@torch.no_grad()
def svdd_score(net, loader, c, device):
    net.eval()
    scores = []
    for batch in loader:
        xb = batch[0].to(device)
        z = net(xb)
        dist = torch.sum((z - c)**2, dim=1)
        scores.append(dist.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def calibrate_threshold_by_validation(scores_val, yval, quantiles=None):
    if quantiles is None:
        quantiles = np.unique(np.r_[np.linspace(0.80, 0.9999, 50),
                                   np.linspace(0.990, 0.9999, 40)])
    val_norm_mask = (yval == 0)
    base = scores_val[val_norm_mask] if np.any(val_norm_mask) else scores_val

    best_f1_class1, best_thr = -1.0, None # 1. Mude o nome da variável
    for q in quantiles:
        thr = np.quantile(base, q)
        yhat = (scores_val > thr).astype(int)
        f1_class1 = f1_score(yval, yhat, pos_label=1)
        if f1_class1 > best_f1_class1: # 2. Use a nova variável na comparação
            best_f1_class1, best_thr = f1_class1, float(thr) # 3. Atualize a nova variável
            
    print(f"[INFO] Calibração validação: melhor F1-classe 1={best_f1_class1:.6f} @ thr={best_thr:.6e}") # 4. Mude a mensagem
    return best_thr, best_f1_class1 # 5. Retorne a nova variável

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_nrows = os.environ.get("NROWS")
    if env_nrows and env_nrows.isdigit(): NROWS = int(env_nrows)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] NROWS={NROWS} | CHUNKSIZE={CHUNKSIZE} | CACHE={USE_CACHE} | USE_FIXED_R={USE_FIXED_R}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{NROWS or 'ALL'}.parquet")

    # ----- Leitura dos CSVs
    t0 = time.time()
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS, use_engine=CSV_ENGINE)
    print(f"[INFO] Leitura CSV total: {time.time()-t0:.1f}s")

    # ----- Featurização (com cache)
    print("[INFO] Gerando features do treino (rápido) ...")
    X_all = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values
    print(f"[INFO] Features treino: {X_all.shape} | y_all: {y_all.shape}")

    print("[INFO] Estatística de classe no treino:")
    uniq, cnt = np.unique(y_all, return_counts=True)
    for u, c in zip(uniq, cnt): print(f"[INFO]   classe {u}: {c}")

    # ----- Split train/val
    print(f"[INFO] Split estratificado train/val (val_size={VAL_SIZE}) ...")
    Xtr_df, Xval_df, ytr, yval = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"[INFO] Xtrain: {Xtr_df.shape}, Xval: {Xval_df.shape}")
    print(f"[INFO] Contagem classes: train 0={np.sum(ytr==0)},1={np.sum(ytr==1)} | val 0={np.sum(yval==0)},1={np.sum(yval==1)}")

    # ----- Scaler (ajuste só nos normais do treino)
    print("[INFO] Ajustando StandardScaler SOMENTE nos normais do treino ...")
    mask_norm_tr = (ytr == 0)
    scaler = StandardScaler().fit(Xtr_df[mask_norm_tr])
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)

    # ----- DataLoaders
    Xtr_norm = Xtr[mask_norm_tr]
    dl_norm = DataLoader(NPDataset(Xtr_norm), batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
    dl_val  = DataLoader(NPDataset(Xval, yval), batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

    # ----- Modelo e centro
    print("[INFO] Inicializando rede e centro c ...")
    net = DeepSVDDNet(d_in=Xtr.shape[1]).to(device)
    c = init_center_c(net, dl_norm, device)
    print("[INFO] Centro c inicializado.")

    # ----- Warm-up curto
    print(f"[INFO] Warm-up: epochs={WARMUP_EPOCHS} ...")
    warmup_minimize_mean_distance(net, dl_norm, c, epochs=WARMUP_EPOCHS, lr=LR, wd=WD, device=device)

    # ----- R fixo por quantil (1-NU) nos normais
    print("[INFO] Variante FIXED-R: definindo R pelo quantil (1-NU) das distâncias nos normais ...")
    dists_norm = compute_distances_on_loader(net, dl_norm, c, device)
    R2 = float(np.quantile(dists_norm.numpy(), 1.0 - NU))
    R2 = max(R2, 1e-12)
    print(f"[INFO] R^2 (FIXED) = {R2:.6e}")

    # (Opcional: refino, aqui desativado com EPOCHS=0)
    if EPOCHS > 0:
        print(f"[INFO] Refinando embedding (mean distance) por {EPOCHS} épocas ...")
        warmup_minimize_mean_distance(net, dl_norm, c, epochs=EPOCHS, lr=LR, wd=WD, device=device)

    # =========================
    # Treino final + Recalibração + Submissão
    # =========================
    print("\n[INFO] Treino final (todos normais do treino completo) + recalibração e submissão ...")
    print("[INFO] Featurizando teste (com cache) ...")
    Xte_df = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)
    print(f"[INFO] Features teste: {Xte_df.shape}")

    print("[INFO] Ajustando scaler FINAL (somente normais do treino completo) ...")
    mask_norm_all = (y_all == 0)
    scaler_full = StandardScaler().fit(X_all[mask_norm_all])
    X_all_s = scaler_full.transform(X_all)
    X_te_s  = scaler_full.transform(Xte_df)
    X_val_final = scaler_full.transform(Xval_df)   # <--- validação no mesmo scaler do FINAL

    # Treinar rede FINAL com todos normais
    X_all_norm = X_all_s[mask_norm_all]
    dl_norm_full = DataLoader(NPDataset(X_all_norm), batch_size=BATCH_TRAIN, shuffle=True, num_workers=0)
    dl_val_final = DataLoader(NPDataset(X_val_final, yval), batch_size=BATCH_EVAL, shuffle=False, num_workers=0)

    print("[INFO] Reinicializando rede e centro c (treino final) ...")
    net_final = DeepSVDDNet(d_in=X_all_s.shape[1]).to(device)
    c_final = init_center_c(net_final, dl_norm_full, device)

    print(f"[INFO] Warm-up final: epochs={WARMUP_EPOCHS} ...")
    warmup_minimize_mean_distance(net_final, dl_norm_full, c_final, epochs=WARMUP_EPOCHS, lr=LR, wd=WD, device=device)

    print("[INFO] Treino final (FIXED-R): definindo R_final por quantil (1-NU) ...")
    dists_norm_full = compute_distances_on_loader(net_final, dl_norm_full, c_final, device)
    R2_final = float(np.quantile(dists_norm_full.numpy(), 1.0 - NU))
    R2_final = max(R2_final, 1e-12)
    print(f"[INFO] R^2_final (FIXED) = {R2_final:.6e}")

    if EPOCHS > 0:
        print(f"[INFO] Refinando embedding final por {EPOCHS} épocas ...")
        warmup_minimize_mean_distance(net_final, dl_norm_full, c_final, epochs=EPOCHS, lr=LR, wd=WD, device=device)

    # ----- Recalibração do limiar COM O MODELO FINAL -----
    print("[INFO] Recalibrando limiar na validação com o MODELO FINAL ...")
    scores_val_final = svdd_score(net_final, dl_val_final, c_final, device)
    best_thr_final, best_f1_final = calibrate_threshold_by_validation(scores_val_final, yval)
    ypred_val_final = (scores_val_final > best_thr_final).astype(int)

    if np.any(yval==0) and np.any(yval==1):
        print(f"[INFO] F1 macro (val, FINAL): {f1_score(yval, ypred_val_final, average='macro'):.6f}")
        print("[INFO] Classification report (val, FINAL):")
        print(classification_report(yval, ypred_val_final, digits=4))
    else:
        acc = (ypred_val_final == yval).mean()
        print(f"[WARN] Val sem as duas classes. Accuracy (FINAL): {acc:.6f}")

    # ----- Inferência no TESTE com limiar calibrado final -----
    print("[INFO] Gerando predições no teste ...")
    dl_test = DataLoader(NPDataset(X_te_s), batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
    scores_test = svdd_score(net_final, dl_test, c_final, device)
    ypred_test = (scores_test > best_thr_final).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[INFO] Distribuição das predições (teste):")
    for u, c_ in zip(uniq_pred, cnt_pred): print(f"[INFO]   classe {u}: {c_}")
    print(f"[INFO] Limiar usado no teste (calibrado no FINAL) = {best_thr_final:.6e}")

    # ----- Submissão
    print("[INFO] Salvando submission.csv ...")
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission.csv')}")

