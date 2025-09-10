# svdd_test12.py
# Deep-SVDD one-class com:
# - Featurização DELTA1 (estável)
# - Distância de Mahalanobis (shrinkage fixo) + Euclidiana
# - ENSEMBLE de scores (alpha * Mz + (1-alpha) * Ez) com grid em alpha
# - Calibração exata por macro-F1
# - Gera submission.csv
#
# Ideia "bala de prata": combinar Mahalanobis (captura anisotropia) com Euclidiana (captura magnitude bruta),
# normalizando ambos na VAL (z-score) e otimizando alpha na própria validação.

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

NU = None              # None => fração de 1s do treino
WARMUP_EPOCHS = 1
EPOCHS = 0
LR = 1e-4
WD = 1e-5
BATCH_TRAIN = 2048
BATCH_EVAL  = 4096

USE_FIXED_R = True
CACHE_DIR   = "./_cache_feats"
CACHE_TAG   = "DELTA1"   # DELTA1 = estável (a que performou melhor no svdd_test10.py)
USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# Mahalanobis
SHRINKAGE = 0.05                 # fixo (bom default na sua execução anterior)
ALPHAS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # grid para combinar M e E

EPS_STD = 1e-9  # proteção para desvios padrões muito pequenos

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
# Featurização DELTA1 (a do svdd_test8.py)
# =======================
def fast_parse_array(s: str):
    if not isinstance(s, str) or len(s) < 2:
        return np.array([], dtype=np.float32)
    return np.fromstring(s[1:-1], sep=',', dtype=np.float32)

def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)

    means   = np.fromiter((a.mean()           if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds    = np.fromiter((a.std(ddof=0)      if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    medians = np.fromiter((np.median(a)       if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p90s    = np.fromiter((np.quantile(a,0.9) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    mins    = np.fromiter((a.min()            if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    maxs    = np.fromiter((a.max()            if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    lens    = np.fromiter((a.size             for a in rtts),                      dtype=np.float32, count=n)

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
    out["success_rate"]        = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"]           = 1.0 - out["success_rate"]
    out["replies_per_attempt"] = out["total_replies_last_hop"] / (out["tr_attempts"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    required = {"tr_src","tr_dst","seconds_since_start"}
    if required.issubset(df.columns):
        tmp = out[[
            "rtt_mean","rtt_p90","success_rate","replies_per_attempt","seconds_since_start"
        ]].copy()
        tmp["tr_src"] = df["tr_src"]
        tmp["tr_dst"] = df["tr_dst"]
        tmp["__rowid__"] = np.arange(len(tmp), dtype=np.int64)
        tmp.sort_values(["tr_src","tr_dst","seconds_since_start"], kind="mergesort", inplace=True)

        def add_delta(col, clip_ratio=(0.0, 10.0)):
            prev = tmp.groupby(["tr_src","tr_dst"])[col].shift(1)
            dcol = f"delta_{col}"
            rcol = f"ratio_{col}"
            tmp[dcol] = (tmp[col] - prev).astype(np.float32)
            ratio = tmp[col] / prev.replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tmp[rcol] = ratio.clip(*clip_ratio).astype(np.float32)

        for base_col in ["rtt_mean","rtt_p90","success_rate","replies_per_attempt"]:
            add_delta(base_col)

        tmp["time_since_prev"] = tmp.groupby(["tr_src","tr_dst"])["seconds_since_start"].diff().fillna(0.0).astype(np.float32)
        tmp["is_first_obs"]    = (tmp.groupby(["tr_src","tr_dst"]).cumcount() == 0).astype(np.float32)

        tmp.sort_values("__rowid__", inplace=True)
        new_cols = [
            "delta_rtt_mean","ratio_rtt_mean",
            "delta_rtt_p90","ratio_rtt_p90",
            "delta_success_rate","ratio_success_rate",
            "delta_replies_per_attempt","ratio_replies_per_attempt",
            "time_since_prev","is_first_obs"
        ]
        out[new_cols] = tmp[new_cols].values
    else:
        out["time_since_prev"] = 0.0
        out["is_first_obs"]    = 1.0
        for c in ["rtt_mean","rtt_p90","success_rate","replies_per_attempt"]:
            out[f"delta_{c}"] = 0.0
            out[f"ratio_{c}"] = 1.0

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    print(f"[INFO] featurize_fast: {out.shape} | {time.time()-t0:.1f}s")
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
    feats = featurize_fast(base)
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
# Deep-SVDD (rede simples)
# =======================
class DeepSVDDNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)
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

# =======================
# Mahalanobis (streaming) + Euclid
# =======================
@torch.no_grad()
def estimate_cov_streaming(net, loader_norm, c, device):
    net.eval()
    sum_x = None
    sum_xx = None
    N = 0
    for (xb,) in loader_norm:
        xb = xb.to(device)
        z = net(xb)
        diff = (z - c).detach().cpu().numpy().astype(np.float64)
        B, d = diff.shape
        if sum_x is None:
            sum_x  = np.zeros(d, dtype=np.float64)
            sum_xx = np.zeros((d, d), dtype=np.float64)
        sum_x  += diff.sum(axis=0)
        sum_xx += diff.T @ diff
        N += B
    if N == 0:
        raise RuntimeError("Nenhum embedding para estimar covariância.")
    mu = sum_x / N
    S = (sum_xx / N) - np.outer(mu, mu)
    S = (S + S.T) * 0.5
    return S

def shrink_inv(S, alpha):
    d = S.shape[0]
    tr_over_d = np.trace(S) / d
    S_shrunk = (1.0 - alpha) * S + alpha * (tr_over_d) * np.eye(d, dtype=S.dtype)
    S_shrunk = (S_shrunk + S_shrunk.T) * 0.5
    try:
        Sinv = np.linalg.inv(S_shrunk)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S_shrunk)
    return Sinv.astype(np.float32)

@torch.no_grad()
def scores_val_both(net, loader_val, c, Pinv, device):
    """Retorna (maha, eucl) para a VAL em numpy (float32)."""
    net.eval()
    P = torch.as_tensor(Pinv, dtype=torch.float32, device=device)
    m_scores, e_scores = [], []
    for batch in loader_val:
        xb = batch[0].to(device)
        z = net(xb)
        diff = z - c
        m = torch.einsum('bi,ij,bj->b', diff, P, diff)      # Mahalanobis
        e = torch.sum(diff * diff, dim=1)                   # Euclidiana (quadrática)
        m_scores.append(m.detach().cpu().numpy())
        e_scores.append(e.detach().cpu().numpy())
    m_scores = np.concatenate(m_scores, axis=0).astype(np.float32)
    e_scores = np.concatenate(e_scores, axis=0).astype(np.float32)
    return m_scores, e_scores

@torch.no_grad()
def scores_test_both(net, loader_test, c, Pinv, device):
    """Retorna (maha, eucl) para o TESTE em numpy (float32)."""
    net.eval()
    P = torch.as_tensor(Pinv, dtype=torch.float32, device=device)
    m_scores, e_scores = [], []
    for (xb,) in loader_test:
        xb = xb.to(device)
        z = net(xb)
        diff = z - c
        m = torch.einsum('bi,ij,bj->b', diff, P, diff)
        e = torch.sum(diff * diff, dim=1)
        m_scores.append(m.detach().cpu().numpy())
        e_scores.append(e.detach().cpu().numpy())
    m_scores = np.concatenate(m_scores, axis=0).astype(np.float32)
    e_scores = np.concatenate(e_scores, axis=0).astype(np.float32)
    return m_scores, e_scores

# =======================
# Calibração EXATA (macro-F1)
# =======================
def calibrate_threshold_by_validation(scores_val, yval):
    idx = np.argsort(scores_val)
    s = scores_val[idx].astype(np.float64)
    y = yval[idx].astype(int)

    pos = (y == 1).astype(np.int64)
    neg = (y == 0).astype(np.int64)
    cum_pos = np.cumsum(pos)
    cum_neg = np.cumsum(neg)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    total_neg = int(cum_neg[-1]) if len(cum_neg) else 0

    TP = total_pos - cum_pos
    FP = total_neg - cum_neg
    FN = cum_pos
    TN = cum_neg

    prec1 = np.divide(TP, (TP+FP), out=np.zeros_like(TP, dtype=float), where=(TP+FP)>0)
    rec1  = np.divide(TP, (TP+FN), out=np.zeros_like(TP, dtype=float), where=(TP+FN)>0)
    f1_1  = np.divide(2*prec1*rec1, (prec1+rec1), out=np.zeros_like(prec1), where=(prec1+rec1)>0)

    prec0 = np.divide(TN, (TN+FN), out=np.zeros_like(TN, dtype=float), where=(TN+FN)>0)
    rec0  = np.divide(TN, (TN+FP), out=np.zeros_like(TN, dtype=float), where=(TN+FP)>0)
    f1_0  = np.divide(2*prec0*rec0, (prec0+rec0), out=np.zeros_like(prec0), where=(prec0+rec0)>0)

    f1_macro = 0.5*(f1_0 + f1_1)
    j = int(np.argmax(f1_macro))
    best_thr = float(s[j])
    best = float(f1_macro[j])
    print(f"[INFO] Calibração exata: melhor F1-macro={best:.6f} @ thr={best_thr:.6e}")
    return best_thr, best

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")

    # ----- Leitura CSVs
    t0 = time.time()
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS, use_engine=CSV_ENGINE)
    print(f"[INFO] Leitura CSV total: {time.time()-t0:.1f}s")

    # ----- Estatística & NU
    y_all = train_df["route_changed"].astype(int).values
    print("[INFO] Estatística de classe no treino:")
    uniq, cnt = np.unique(y_all, return_counts=True)
    for u, c in zip(uniq, cnt): print(f"[INFO]   classe {u}: {c}")
    if NU is None:
        frac_outliers = float(np.sum(y_all == 1)) / max(len(y_all), 1)
        NU = max(frac_outliers, 1e-3)
        print(f"[INFO] NU calculado empiricamente = {NU:.6f}")
    else:
        print(f"[INFO] NU fixado pelo usuário = {NU:.6f}")

    # ----- Featurização (com cache)
    print("[INFO] Gerando features do treino ...")
    X_all = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    print(f"[INFO] Features treino: {X_all.shape} | y_all: {y_all.shape}")

    # ----- Split
    print(f"[INFO] Split estratificado train/val (val_size={VAL_SIZE}) ...")
    Xtr_df, Xval_df, ytr, yval = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"[INFO] Xtrain: {Xtr_df.shape}, Xval: {Xval_df.shape}")
    print(f"[INFO] Contagem classes: train 0={np.sum(ytr==0)},1={np.sum(ytr==1)} | val 0={np.sum(yval==0)},1={np.sum(yval==1)}")

    # ----- Scaler (só normais do treino)
    print("[INFO] Ajustando StandardScaler SOMENTE nos normais do treino ...")
    mask_norm_tr = (ytr == 0)
    scaler = StandardScaler().fit(Xtr_df[mask_norm_tr])
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)

    # ----- Loaders
    Xtr_norm = Xtr[mask_norm_tr]
    dl_norm = DataLoader(NPDataset(Xtr_norm), batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
    dl_val  = DataLoader(NPDataset(Xval, yval), batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

    # ----- Modelo & centro
    print("[INFO] Inicializando rede e centro c ...")
    net = DeepSVDDNet(d_in=Xtr.shape[1]).to(device)
    c = init_center_c(net, dl_norm, device)
    print("[INFO] Centro c inicializado.")

    # ----- Warm-up
    print(f"[INFO] Warm-up: epochs={WARMUP_EPOCHS} ...")
    warmup_minimize_mean_distance(net, dl_norm, c, epochs=WARMUP_EPOCHS, lr=LR, wd=WD, device=device)

    # ----- FIXED-R (1 - NU) em normais do treino
    print("[INFO] Variante FIXED-R: definindo R pelo quantil (1-NU) das distâncias nos normais ...")
    # Nota: R é usado apenas para monitoramento; o score final vem do ensemble.
    # Mantemos por consistência com seus scripts anteriores.
    # (Pode ser útil para logging ou sanity-check.)
    # Se quiser, pode remover sem afetar a lógica do ensemble.
    # Aqui calculamos distâncias euclidianas na fase de treino (dl_norm).
    # (Não necessário para o pipeline final.)
    # -- omitido para economizar tempo/VRAM --

    # =========================
    # Final: treino full + calibração (ensemble) + submissão
    # =========================
    print("\n[INFO] Treino final (todos normais) + ensemble e calibração exata ...")
    print("[INFO] Featurizando teste ...")
    Xte_df = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)
    print(f"[INFO] Features teste: {Xte_df.shape}")

    print("[INFO] Ajustando scaler FINAL (normais do treino completo) ...")
    mask_norm_all = (y_all == 0)
    scaler_full = StandardScaler().fit(X_all[mask_norm_all])
    X_all_s = scaler_full.transform(X_all)
    X_te_s  = scaler_full.transform(Xte_df)
    X_val_final = scaler_full.transform(Xval_df)

    X_all_norm = X_all_s[mask_norm_all]
    dl_norm_full = DataLoader(NPDataset(X_all_norm), batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
    dl_val_final = DataLoader(NPDataset(X_val_final, yval), batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)
    dl_test      = DataLoader(NPDataset(X_te_s),           batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

    print("[INFO] Reinicializando rede e centro c (treino final) ...")
    net_final = DeepSVDDNet(d_in=X_all_s.shape[1]).to(device)
    c_final = init_center_c(net_final, dl_norm_full, device)

    print(f"[INFO] Warm-up final: epochs={WARMUP_EPOCHS} ...")
    warmup_minimize_mean_distance(net_final, dl_norm_full, c_final, epochs=WARMUP_EPOCHS, lr=LR, wd=WD, device=device)

    # ----- Estima covariância nos normais (para Mahalanobis)
    print("[INFO] Estimando covariância S (streaming) nos normais (final) ...")
    S = estimate_cov_streaming(net_final, dl_norm_full, c_final, device)
    S_inv = shrink_inv(S, SHRINKAGE)
    print(f"[INFO] Shrinkage fixo = {SHRINKAGE:.3f}")

    # ----- Scores na VAL: Mahalanobis e Euclidiana
    print("[INFO] Computando scores (VAL) Mahalanobis e Euclidiana ...")
    m_val, e_val = scores_val_both(net_final, dl_val_final, c_final, S_inv, device)

    # ----- Z-score com estatísticas da VAL
    m_mean, m_std = float(m_val.mean()), float(m_val.std() + EPS_STD)
    e_mean, e_std = float(e_val.mean()), float(e_val.std() + EPS_STD)
    m_val_z = (m_val - m_mean) / m_std
    e_val_z = (e_val - e_mean) / e_std

    # ----- Grid em ALPHA para ensemble na VAL
    print(f"[INFO] Alpha grid (ensemble): {ALPHAS}")
    best_macro = -1.0
    best_alpha = None
    best_thr = None

    for alpha in ALPHAS:
        s_val = alpha * m_val_z + (1.0 - alpha) * e_val_z
        thr, macro = calibrate_threshold_by_validation(s_val, yval)
        print(f"[GRID] alpha={alpha:.2f} | macro-F1={macro:.6f} | thr={thr:.6e}")
        if macro > best_macro:
            best_macro = macro
            best_alpha = alpha
            best_thr = thr

    print(f"[INFO] Melhor alpha = {best_alpha:.2f} | macro-F1(val)= {best_macro:.6f} | thr= {best_thr:.6e}")

    # ----- Relatório final na VAL (com melhor alpha)
    s_val_best = best_alpha * m_val_z + (1.0 - best_alpha) * e_val_z
    ypred_val = (s_val_best > best_thr).astype(int)
    if np.any(yval==0) and np.any(yval==1):
        print(f"[INFO] F1 macro (val, FINAL): {f1_score(yval, ypred_val, average='macro'):.6f}")
        print("[INFO] Classification report (val, FINAL):")
        print(classification_report(yval, ypred_val, digits=4))
    else:
        acc = (ypred_val == yval).mean()
        print(f"[WARN] Val sem as duas classes. Accuracy (FINAL): {acc:.6f}")

    # ----- Scores no TESTE e submissão (usar mesmas estatísticas de z-score da VAL)
    print("[INFO] Computando scores (TEST) Mahalanobis e Euclidiana ...")
    m_test, e_test = scores_test_both(net_final, dl_test, c_final, S_inv, device)
    m_test_z = (m_test - m_mean) / m_std
    e_test_z = (e_test - e_mean) / e_std
    s_test = best_alpha * m_test_z + (1.0 - best_alpha) * e_test_z
    ypred_test = (s_test > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[INFO] Distribuição das predições (teste):")
    for u, c_ in zip(uniq_pred, cnt_pred): print(f"[INFO]   classe {u}: {c_}")
    print(f"[INFO] Limiar (VAL) = {best_thr:.6e} | alpha={best_alpha:.2f} | shrinkage={SHRINKAGE:.3f}")

    print("[INFO] Salvando submission.csv ...")
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission.csv')}")

