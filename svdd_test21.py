# svdd_mlab_oneclass_gpu_calibrated_final.py
# Deep-SVDD one-class (FIXED-R + calibração) para mudança de rota (M-Lab)
# - Featurização avançada e em cache
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
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from scipy.stats import kurtosis, skew

# =======================
# Configurações principais
# =======================
SEED = 42
VAL_SIZE = 0.20

NU = 0.10  # Aumentado para evitar colapso
WARMUP_EPOCHS = 1
EPOCHS = 10  # Aumentado para permitir refino
LR = 1e-3
WD = 1e-6  # Diminuído para evitar colapso
BATCH_TRAIN = 2048
BATCH_EVAL = 4096

USE_FIXED_R = True

CACHE_DIR = "./_cache_feats_ADVANCED"
USE_CACHE = True
CHUNKSIZE = None
CSV_ENGINE = "pyarrow"
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
NROWS = None

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
# Featurização RÁPIDA (COM NOVOAS FEATURES)
# =======================
def fast_parse_array(s: str):
    if not isinstance(s, str) or len(s) < 2:
        return np.array([], dtype=np.float32)
    return np.fromstring(s[1:-1], sep=',', dtype=np.float32)

def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)

    means = np.fromiter((a.mean() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds = np.fromiter((a.std(ddof=0) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    medians = np.fromiter((np.median(a) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p90s = np.fromiter((np.quantile(a, 0.9) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p95s = np.fromiter((np.quantile(a, 0.95) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    mins = np.fromiter((a.min() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    maxs = np.fromiter((a.max() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    lens = np.fromiter((a.size for a in rtts), dtype=np.float32, count=n)
    rtt_mad = np.fromiter((np.median(np.abs(a - np.median(a))) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    rtt_kurtosis = np.fromiter((kurtosis(a) if a.size >= 4 else 0.0 for a in rtts), dtype=np.float32, count=n)
    rtt_skewness = np.fromiter((skew(a) if a.size >= 3 else 0.0 for a in rtts), dtype=np.float32, count=n)

    date_index_series = df["date_index"] if "date_index" in df.columns else pd.Series(0, index=df.index)

    out = pd.DataFrame({
        "rtt_mean": means, "rtt_std": stds, "rtt_median": medians,
        "rtt_p90": p90s, "rtt_p95": p95s, "rtt_min": mins, "rtt_max": maxs, "rtt_len": lens,
        "rtt_mad": rtt_mad,
        "rtt_range": maxs - mins,
        "rtt_kurtosis": rtt_kurtosis,
        "rtt_skewness": rtt_skewness,
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

def train_model(net, dl_norm, c, R2, epochs=1, lr=1e-3, wd=1e-5, device="cuda"):
    net.to(device).train()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, epochs+1):
        t0, ep_loss = time.time(), 0.0
        for (xb,) in dl_norm:
            xb = xb.to(device)
            z = net(xb)
            dist_sq = torch.sum((z - c)**2, dim=1)
            
            loss = R2 + (1/NU) * torch.mean(torch.clamp(dist_sq - R2, min=0))
            
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        print(f"[INFO] Treino {ep:02d}/{epochs} | mean_loss={ep_loss/len(dl_norm):.6f} | {time.time()-t0:.1f}s")

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

# =======================
# Metodo de Calibracao e Ensemble
# =======================
def calibrate_threshold_by_validation(scores_val, yval, quantiles=None):
    if quantiles is None:
        quantiles = np.unique(np.r_[np.linspace(0.80, 0.9999, 50),
                                   np.linspace(0.990, 0.9999, 40)])
    val_norm_mask = (yval == 0)
    base = scores_val[val_norm_mask] if np.any(val_norm_mask) else scores_val

    best_f1_class1, best_thr = -1.0, None
    for q in quantiles:
        thr = np.quantile(base, q)
        yhat = (scores_val > thr).astype(int)
        f1_class1 = f1_score(yval, yhat, pos_label=1)
        if f1_class1 > best_f1_class1:
            best_f1_class1, best_thr = f1_class1, float(thr)
            
    print(f"[INFO] Calibração validação: melhor F1-classe 1={best_f1_class1:.6f} @ thr={best_thr:.6e}")
    return best_thr, best_f1_class1

def calibrate_and_evaluate_stacking(scores_val_svdd, scores_val_lgbm, yval):
    meta_df_val = pd.DataFrame({
        "svdd_score": scores_val_svdd,
        "lgbm_score": scores_val_lgbm
    })
    
    X_meta_val = meta_df_val.values
    
    meta_model = lgb.LGBMClassifier(random_state=SEED, n_estimators=50)
    meta_model.fit(X_meta_val, yval)
    meta_scores_val = meta_model.predict_proba(X_meta_val)[:, 1]
    
    best_f1_class1, best_thr = calibrate_threshold_by_validation(meta_scores_val, yval)
    
    ypred_val = (meta_scores_val > best_thr).astype(int)
    
    print("\n[RESULTADO FINAL] Classification report (val, STACKING ENSEMBLE):")
    print(classification_report(yval, ypred_val, digits=4))
    
    return meta_model, best_thr

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_nrows = os.environ.get("NROWS")
    if env_nrows and env_nrows.isdigit(): NROWS = int(env_nrows)
    print("[MAIN] Iniciando pipeline de dados...")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_ADVANCED_{NROWS or 'ALL'}.parquet")
    cache_test = os.path.join(CACHE_DIR, f"test_feats_ADVANCED_{NROWS or 'ALL'}.parquet")

    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    test_df = read_csv_fast(TEST_CSV, nrows=NROWS, use_engine=CSV_ENGINE)

    X_all = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values
    
    X_te_df = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)
    
    print("[MAIN] Realizando split, SMOTE e scaling...")
    Xtr_df, Xval_df, ytr, yval = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"[INFO] Contagem classes: train 0={np.sum(ytr==0)},1={np.sum(ytr==1)} | val 0={np.sum(yval==0)},1={np.sum(yval==1)}")
    
    scaler = StandardScaler().fit(Xtr_df)
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    Xte = scaler.transform(X_te_df)
    
    smote = SMOTE(random_state=SEED)
    print("[MAIN] Aplicando SMOTE no conjunto de treino...")
    Xtr_smote, ytr_smote = smote.fit_resample(Xtr, ytr)

    # ==============================================================================
    # >>> FASE 1: Treinando Modelo Base DEEP SVDD <<<
    # ==============================================================================
    print("\n" + "=" * 80)
    print(">>> FASE 1: Treinando Modelo Base DEEP SVDD <<<")
    print("=" * 80)

    Xtr_norm_smote = Xtr_smote[ytr_smote == 0]
    dl_norm_smote = DataLoader(NPDataset(Xtr_norm_smote), batch_size=BATCH_TRAIN, shuffle=True)
    dl_val = DataLoader(NPDataset(Xval), batch_size=BATCH_EVAL)
    dl_te = DataLoader(NPDataset(Xte), batch_size=BATCH_EVAL)

    net = DeepSVDDNet(d_in=Xtr_smote.shape[1]).to(device)
    c = init_center_c(net, dl_norm_smote, device)
    
    # Warm-up (perda de distância simples para inicialização)
    train_model(net, dl_norm_smote, c, R2=0, epochs=WARMUP_EPOCHS, lr=LR, wd=WD, device=device)
    
    # Treino com perda Deep-SVDD (Fixed-R)
    dists_norm_smote = compute_distances_on_loader(net, dl_norm_smote, c, device)
    R2 = float(np.quantile(dists_norm_smote.numpy(), 1.0 - NU))
    R2 = max(R2, 1e-12)

    if EPOCHS > 0:
        train_model(net, dl_norm_smote, c, R2, epochs=EPOCHS, lr=LR, wd=WD, device=device)

    scores_val_svdd = svdd_score(net, dl_val, c, device)
    scores_te_svdd = svdd_score(net, dl_te, c, device)
    
    print("[SVDD] Scores de validação e teste gerados com sucesso.")

    # ==============================================================================
    # >>> FASE 2: Treinando Modelo Base LIGHTGBM <<<
    # ==============================================================================
    print("\n" + "=" * 80)
    print(">>> FASE 2: Treinando Modelo Base LIGHTGBM <<<")
    print("=" * 80)
    lgbm = lgb.LGBMClassifier(random_state=SEED, n_estimators=2500, learning_rate=0.01)
    lgbm.fit(Xtr_smote, ytr_smote,
             eval_set=[(Xval, yval)],
             eval_metric='auc',
             callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])

    scores_val_lgbm = lgbm.predict_proba(Xval)[:, 1]
    scores_te_lgbm = lgbm.predict_proba(Xte)[:, 1]

    print("[LGBM] Scores de validação e teste gerados com sucesso.")

    # ==============================================================================
    # >>> FASE 3: Treinando Meta-Modelo de Stacking <<<
    # ==============================================================================
    print("\n" + "=" * 80)
    print(">>> FASE 3: Treinando Meta-Modelo de Stacking <<<")
    print("=" * 80)
    
    meta_model, best_thr = calibrate_and_evaluate_stacking(scores_val_svdd, scores_val_lgbm, yval)
    
    print("[MAIN] Gerando predições finais no conjunto de teste...")
    meta_df_te = pd.DataFrame({
        "svdd_score": scores_te_svdd,
        "lgbm_score": scores_te_lgbm
    })
    
    X_meta_te = meta_df_te.values
    final_scores_te = meta_model.predict_proba(X_meta_te)[:, 1]
    ypred_test = (final_scores_te > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[MAIN] Distribuição das predições finais (teste):")
    for u, c in zip(uniq_pred, cnt_pred): print(f"[MAIN]    classe {u}: {c}")

    print("[MAIN] Salvando submission_stacking.csv...")
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    sub.to_csv("submission_stacking.csv", index=False)
    print(f"[MAIN] Arquivo salvo: {os.path.abspath('submission_stacking.csv')}")

    print("\n[MAIN] Processo concluído com sucesso!")
