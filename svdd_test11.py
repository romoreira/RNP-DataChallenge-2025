# svdd_test11.py
# Deep-SVDD one-class (FIXED-R + calibração exata) com:
# - Featurização DELTA2 (enriquecida: std/median, *_per_sec, prev3)
# - Distância de Mahalanobis com shrinkage
# - GRID de SHRINKAGE na validação para maximizar macro-F1
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

NU = None              # None => usa fração de 1s do treino
WARMUP_EPOCHS = 1
EPOCHS = 0
LR = 1e-4
WD = 1e-5
BATCH_TRAIN = 2048
BATCH_EVAL  = 4096

USE_FIXED_R = True
CACHE_DIR   = "./_cache_feats"
CACHE_TAG   = "DELTA2"   # DELTA2 = features enriquecidas
USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# Mahalanobis
USE_MAHALANOBIS = True
SHRINK_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]  # grid simples
VAL_CHUNK = 1_000_000  # tamanho do chunk para computar scores na validação em CPU

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
# Featurização DELTA2 (enriquecida)
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
        base_cols = ["rtt_mean","rtt_p90","success_rate","replies_per_attempt","rtt_std","rtt_median"]
        tmp = out[base_cols + ["seconds_since_start"]].copy()
        tmp["tr_src"] = df["tr_src"]
        tmp["tr_dst"] = df["tr_dst"]
        tmp["__rowid__"] = np.arange(len(tmp), dtype=np.int64)

        tmp.sort_values(["tr_src","tr_dst","seconds_since_start"],
                        kind="mergesort", inplace=True)

        tmp["time_since_prev"] = tmp.groupby(["tr_src","tr_dst"])["seconds_since_start"] \
                                   .diff().fillna(0.0).astype(np.float32)
        tmp["is_first_obs"] = (tmp.groupby(["tr_src","tr_dst"]).cumcount() == 0).astype(np.float32)

        def add_delta_and_ratio(col, clip_ratio=(0.0, 10.0)):
            prev1 = tmp.groupby(["tr_src","tr_dst"])[col].shift(1)
            dcol  = f"delta_{col}"
            rcol  = f"ratio_{col}"
            tmp[dcol] = (tmp[col] - prev1).astype(np.float32)
            ratio = tmp[col] / prev1.replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tmp[rcol] = ratio.clip(*clip_ratio).astype(np.float32)
            tmp[f"{dcol}_per_sec"] = (tmp[dcol] / (tmp["time_since_prev"] + 1.0)).astype(np.float32)

        def add_dev_vs_prev3(col, clip_ratio=(0.0, 10.0)):
            g = tmp.groupby(["tr_src","tr_dst"])[col]
            p1, p2, p3 = g.shift(1), g.shift(2), g.shift(3)
            cnt = (~p1.isna()).astype(np.int16) + (~p2.isna()).astype(np.int16) + (~p3.isna()).astype(np.int16)
            avg_prev3 = (p1.fillna(0) + p2.fillna(0) + p3.fillna(0)) / cnt.replace(0, np.nan)
            avg_prev3 = avg_prev3.fillna(p1).fillna(0.0)
            d3  = f"dev_vs_prev3_{col}"
            r3  = f"ratio_vs_prev3_{col}"
            tmp[d3] = (tmp[col] - avg_prev3).astype(np.float32)
            ratio3 = tmp[col] / avg_prev3.replace(0, np.nan)
            ratio3 = ratio3.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tmp[r3] = ratio3.clip(*clip_ratio).astype(np.float32)
            tmp[f"{d3}_per_sec"] = (tmp[d3] / (tmp["time_since_prev"] + 1.0)).astype(np.float32)

        for col in base_cols:
            add_delta_and_ratio(col)
            add_dev_vs_prev3(col)

        tmp.sort_values("__rowid__", inplace=True)

        new_cols = []
        for col in base_cols:
            new_cols += [
                f"delta_{col}", f"ratio_{col}", f"delta_{col}_per_sec",
                f"dev_vs_prev3_{col}", f"ratio_vs_prev3_{col}", f"dev_vs_prev3_{col}_per_sec"
            ]
        new_cols += ["time_since_prev","is_first_obs"]

        out[new_cols] = tmp[new_cols].values
    else:
        out["time_since_prev"] = 0.0
        out["is_first_obs"]    = 1.0
        for c in ["rtt_mean","rtt_p90","success_rate","replies_per_attempt","rtt_std","rtt_median"]:
            out[f"delta_{c}"] = 0.0
            out[f"ratio_{c}"] = 1.0
            out[f"delta_{c}_per_sec"] = 0.0
            out[f"dev_vs_prev3_{c}"] = 0.0
            out[f"ratio_vs_prev3_{c}"] = 1.0
            out[f"dev_vs_prev3_{c}_per_sec"] = 0.0

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

# =======================
# Mahalanobis (streaming + shrinkage)
# =======================
@torch.no_grad()
def estimate_cov_streaming(net, loader_norm, c, device):
    """
    Retorna a covariância S (dxd) dos embeddings centralizados x = (z - c).
    Usa uma passada única sem acumular todos os vetores.
    """
    net.eval()
    sum_x = None
    sum_xx = None
    N = 0
    for (xb,) in loader_norm:
        xb = xb.to(device)
        z = net(xb)
        diff = (z - c).detach().cpu().numpy().astype(np.float64)  # [B, d]
        B, d = diff.shape
        if sum_x is None:
            sum_x  = np.zeros(d, dtype=np.float64)
            sum_xx = np.zeros((d, d), dtype=np.float64)
        sum_x  += diff.sum(axis=0)
        sum_xx += diff.T @ diff
        N += B
    if N == 0:
        raise RuntimeError("Nenhum embedding disponível para estimar a covariância.")
    mu = sum_x / N
    S = (sum_xx / N) - np.outer(mu, mu)  # MLE
    S = (S + S.T) * 0.5
    return S

def shrink_and_inv(S, alpha):
    """S_shrunk = (1-alpha) S + alpha * (tr(S)/d) I; retorna S_inv (float32)."""
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
def embed_diffs_and_labels(net, loader, c, device):
    """
    Retorna (diffs, labels) para TODO o loader em CPU (float32).
    diffs: [N, d], labels: [N]
    """
    net.eval()
    diffs = []
    ys = []
    for batch in loader:
        xb = batch[0].to(device)
        z = net(xb)
        diff = (z - c).detach().cpu().numpy().astype(np.float32)
        diffs.append(diff)
        if len(batch) > 1:
            ys.append(batch[1].detach().cpu().numpy().astype(np.int64))
    diffs = np.concatenate(diffs, axis=0) if diffs else np.zeros((0, 32), dtype=np.float32)
    labels = np.concatenate(ys, axis=0) if ys else None
    return diffs, labels

def maha_scores_from_diffs(diffs, Pinv, chunk=1_000_000):
    """
    Computa m = (x^T Pinv x) em chunks para economia de memória.
    diffs: [N, d] float32; Pinv: [d, d] float32
    """
    N = diffs.shape[0]
    out = np.empty(N, dtype=np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        X = diffs[i:j]                                  # [B, d]
        XP = X @ Pinv                                   # [B, d]
        out[i:j] = np.einsum("bi,bi->b", XP, X)         # diag(XP * X^T)
    return out

@torch.no_grad()
def svdd_score_mahalanobis_loader(net, loader, c, Pinv, device):
    """
    Score por Mahalanobis direto do loader (GPU), útil para o TESTE.
    """
    net.eval()
    P = torch.as_tensor(Pinv, dtype=torch.float32, device=device)
    scores = []
    for (xb,) in loader:
        xb = xb.to(device)
        z = net(xb)
        diff = z - c
        m = torch.einsum('bi,ij,bj->b', diff, P, diff)
        scores.append(m.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=np.float32)

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
    dists_norm = compute_distances_on_loader(net, dl_norm, c, device)
    R2 = float(np.quantile(dists_norm.numpy(), 1.0 - NU))
    R2 = max(R2, 1e-12)
    print(f"[INFO] R^2 (FIXED) = {R2:.6e}")

    if EPOCHS > 0:
        print(f"[INFO] Refinando embedding por {EPOCHS} épocas ... (desativado)")

    # =========================
    # Final: treino full + calibração exata + submissão
    # =========================
    print("\n[INFO] Treino final (todos normais) + recalibração exata (grid shrinkage) e submissão ...")
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

    print("[INFO] Treino final (FIXED-R): definindo R_final por quantil (1-NU) ...")
    dists_norm_full = compute_distances_on_loader(net_final, dl_norm_full, c_final, device)
    R2_final = float(np.quantile(dists_norm_full.numpy(), 1.0 - NU))
    R2_final = max(R2_final, 1e-12)
    print(f"[INFO] R^2_final (FIXED) = {R2_final:.6e}")

    # ----- Estima S (cov dos normais) uma única vez -----
    print("[INFO] Estimando covariância S (streaming) nos normais (final) ...")
    S = estimate_cov_streaming(net_final, dl_norm_full, c_final, device)

    # ----- Embeddings (diffs) de VAL uma única vez -----
    print("[INFO] Extraindo diffs da VAL para varrer shrinkage ...")
    val_diffs, yval_ck = embed_diffs_and_labels(net_final, dl_val_final, c_final, device)
    if yval_ck is None:
        raise RuntimeError("Validação sem rótulos — necessário para calibrar macro-F1.")
    assert yval_ck.shape[0] == val_diffs.shape[0], "Inconsistência entre diffs e labels da validação."
    print(f"[INFO] VAL diffs shape: {val_diffs.shape}")

    # ----- Grid de SHRINKAGE na validação -----
    print(f"[INFO] Shrinkage grid: {SHRINK_GRID}")
    best_macro = -1.0
    best_thr = None
    best_alpha = None
    best_Sinv = None

    for alpha in SHRINK_GRID:
        tgs = time.time()
        Sinv = shrink_and_inv(S, alpha)
        scores_val = maha_scores_from_diffs(val_diffs, Sinv, chunk=VAL_CHUNK)
        thr, macro = calibrate_threshold_by_validation(scores_val, yval_ck)
        print(f"[GRID] alpha={alpha:.3f} | macro-F1={macro:.6f} | thr={thr:.6e} | time={time.time()-tgs:.1f}s")
        if macro > best_macro:
            best_macro = macro
            best_thr = thr
            best_alpha = alpha
            best_Sinv = Sinv

    print(f"[INFO] Melhor shrinkage= {best_alpha:.3f} | macro-F1(val)= {best_macro:.6f} | thr= {best_thr:.6e}")

    # ----- Relatório final na VAL (com melhor alpha) -----
    ypred_val_final = (maha_scores_from_diffs(val_diffs, best_Sinv, chunk=VAL_CHUNK) > best_thr).astype(int)
    if np.any(yval_ck==0) and np.any(yval_ck==1):
        print(f"[INFO] F1 macro (val, FINAL): {f1_score(yval_ck, ypred_val_final, average='macro'):.6f}")
        print("[INFO] Classification report (val, FINAL):")
        print(classification_report(yval_ck, ypred_val_final, digits=4))
    else:
        acc = (ypred_val_final == yval_ck).mean()
        print(f"[WARN] Val sem as duas classes. Accuracy (FINAL): {acc:.6f}")

    # ----- Inferência no TESTE (uma passada, com melhor alpha) -----
    print("[INFO] Gerando predições no teste (Mahalanobis, melhor shrinkage) ...")
    scores_test = svdd_score_mahalanobis_loader(net_final, dl_test, c_final, best_Sinv, device)
    ypred_test = (scores_test > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[INFO] Distribuição das predições (teste):")
    for u, c_ in zip(uniq_pred, cnt_pred): print(f"[INFO]   classe {u}: {c_}")
    print(f"[INFO] Limiar usado no teste (calibrado no FINAL) = {best_thr:.6e} | shrinkage={best_alpha:.3f}")

    # ----- Submissão
    print("[INFO] Salvando submission.csv ...")
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission.csv')}")

