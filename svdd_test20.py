# stacking_ensemble_final.py
# Combina o poder do Deep SVDD e do LightGBM usando um meta-modelo de Stacking.
# 1. Treina os modelos base (SVDD, LGBM).
# 2. Usa suas predições como features para um meta-modelo (Logistic Regression).
# 3. Gera a predição final combinada.

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

# =======================
# Configurações Globais
# =======================
SEED = 42
VAL_SIZE = 0.20
USE_SMOTE = True

# --- Configs de IO e Cache ---
CACHE_DIR   = "./_cache_feats"
CACHE_TAG   = "DELTA1"
USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# --- Configs de Pós-processamento ---
MIN_GROUP_NORMALS = 100
EPS_STD           = 1e-8

# --- Configs do Modelo SVDD ---
SVDD_EPOCHS = 15
SVDD_LR = 1e-4
SVDD_WD = 1e-5
SVDD_BATCH_TRAIN = 2048
SVDD_BATCH_EVAL  = 4096
SVDD_R_QUANTILE = 0.95
SVDD_SHRINKAGE = 0.05

# --- Configs do Modelo LGBM ---
LGBM_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 2500, 'learning_rate': 0.01, 'num_leaves': 40,
    'max_depth': 10, 'seed': SEED, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}

# ================================================================
# Funções de Preparação de Dados (sem alterações)
# ================================================================
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# (As funções read_csv_fast, fast_parse_array, featurize_fast, maybe_cached_features
# são omitidas por brevidade, mas devem ser coladas aqui do script anterior.
# Elas são idênticas.)
# INÍCIO - COLE AS FUNÇÕES DE DADOS AQUI
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
def fast_parse_array(s: str):
    if not isinstance(s, str) or len(s) < 2: return np.array([], dtype=np.float32)
    return np.fromstring(s[1:-1], sep=',', dtype=np.float32)
def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)
    means   = np.fromiter((a.mean()         if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds    = np.fromiter((a.std(ddof=0)    if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    medians = np.fromiter((np.median(a)     if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p90s    = np.fromiter((np.quantile(a,0.9) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    mins    = np.fromiter((a.min()          if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    maxs    = np.fromiter((a.max()          if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    lens    = np.fromiter((a.size           for a in rtts),                   dtype=np.float32, count=n)
    date_index_series = df["date_index"] if "date_index" in df.columns else pd.Series(0, index=df.index)
    out = pd.DataFrame({"rtt_mean": means, "rtt_std": stds, "rtt_median": medians, "rtt_p90": p90s, "rtt_min": mins, "rtt_max": maxs, "rtt_len": lens, "tr_attempts": df["tr_attempts"].astype(np.float32), "total_probes_sent": df["total_probes_sent"].astype(np.float32), "total_replies_last_hop": df["total_replies_last_hop"].astype(np.float32), "seconds_since_start": df["seconds_since_start"].astype(np.float32), "date_index": date_index_series.astype(np.float32),})
    eps = 1e-9
    out["success_rate"]         = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"]            = 1.0 - out["success_rate"]
    out["replies_per_attempt"] = out["total_replies_last_hop"] / (out["tr_attempts"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    required = {"tr_src","tr_dst","seconds_since_start"}
    if required.issubset(df.columns):
        tmp = out[["rtt_mean","rtt_p90","success_rate","replies_per_attempt","seconds_since_start"]].copy()
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
        for base_col in ["rtt_mean","rtt_p90","success_rate","replies_per_attempt"]: add_delta(base_col)
        tmp["time_since_prev"] = tmp.groupby(["tr_src","tr_dst"])["seconds_since_start"].diff().fillna(0.0).astype(np.float32)
        tmp["is_first_obs"]    = (tmp.groupby(["tr_src","tr_dst"]).cumcount() == 0).astype(np.float32)
        tmp.sort_values("__rowid__", inplace=True)
        new_cols = ["delta_rtt_mean","ratio_rtt_mean","delta_rtt_p90","ratio_rtt_p90","delta_success_rate","ratio_success_rate","delta_replies_per_attempt","ratio_replies_per_attempt","time_since_prev","is_first_obs"]
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
            else: print("[INFO] Cache existe mas nrows difere; recalculando features.")
        except Exception as e: print(f"[WARN] Falha ao ler cache ({e}). Recriando.")
    base = read_csv_fast(csv_path, nrows=nrows, use_engine=CSV_ENGINE)
    feats = featurize_fast(base)
    if USE_CACHE:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats.to_parquet(cache_path, index=False)
        print(f"[INFO] Cache salvo em: {cache_path}")
    return feats
# FIM - COLE AS FUNÇÕES DE DADOS AQUI

# ================================================================
# Funções de Pós-processamento e Calibração (sem alterações)
# ================================================================
def make_route_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["tr_src"].astype(np.int64).values << 32) ^ df["tr_dst"].astype(np.int64).values
def compute_group_norm_stats(scores: np.ndarray, y: np.ndarray, keys: np.ndarray, min_group_normals=100, eps_std=1e-8):
    mask_norm = (y == 0)
    s_norm, k_norm = scores[mask_norm], keys[mask_norm]
    g_mean = float(s_norm.mean())
    g_std = float(s_norm.std() + eps_std)
    df = pd.DataFrame({"k": k_norm, "s": s_norm})
    grp = df.groupby("k")["s"]
    mu, sd, cnt = grp.mean(), grp.std().fillna(0.0), grp.size()
    mu_adj = mu.where(cnt >= min_group_normals, g_mean)
    sd_adj = sd.where((cnt >= min_group_normals) & (sd > 0.0), g_std)
    mu_map, sd_map = mu_adj.to_dict(), sd_adj.to_dict()
    n_groups, n_groups_usable = len(mu_map), int(((cnt >= min_group_normals).sum()))
    print(f"[INFO] Grupos na VAL: {n_groups} | com >= {min_group_normals} normais: {n_groups_usable} | g_std={g_std:.6f}")
    return mu_map, sd_map, g_mean, g_std
def apply_group_zscore(scores: np.ndarray, keys: np.ndarray, mu_map, sd_map, g_mean, g_std) -> np.ndarray:
    ks = pd.Series(keys)
    mu = ks.map(mu_map).astype(float).fillna(g_mean).values
    sd = ks.map(sd_map).astype(float).fillna(g_std).values
    sd = np.where(sd <= 0.0, g_std, sd)
    return ((scores - mu) / sd).astype(np.float32)
def calibrate_threshold_by_validation(scores_val, yval):
    idx = np.argsort(scores_val)
    s, y = scores_val[idx].astype(np.float64), yval[idx].astype(int)
    pos, neg = (y == 1).astype(np.int64), (y == 0).astype(np.int64)
    cum_pos, cum_neg = np.cumsum(pos), np.cumsum(neg)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    TP, FP = total_pos - cum_pos, (len(y) - total_pos) - cum_neg
    FN = cum_pos
    prec1 = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) > 0)
    rec1 = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) > 0)
    f1_1 = np.divide(2 * prec1 * rec1, (prec1 + rec1), out=np.zeros_like(prec1), where=(prec1 + rec1) > 0)
    j = int(np.argmax(f1_1))
    best_thr, best = float(s[j]), float(f1_1[j])
    print(f"[INFO] Calibração (F1-score classe 1): melhor F1-1={best:.6f} @ thr={best_thr:.6e}")
    return best_thr, best

# ================================================================
# SEÇÃO 1: Definições e Lógica do Modelo Deep SVDD
# ================================================================
class NPDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None: return (self.X[i],)
        return self.X[i], self.y[i]
class DeepSVDDNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
    def forward(self, x): return self.net(x)
class SVDDLoss(nn.Module):
    def __init__(self, c, R_squared):
        super().__init__()
        self.c, self.R_squared = c, R_squared
    def forward(self, z, y):
        dist_sq = torch.sum((z - self.c)**2, dim=1)
        loss_normal = dist_sq
        loss_anomaly = torch.maximum(torch.tensor(0.0, device=z.device), self.R_squared - dist_sq)
        loss = torch.where(y == 0, loss_normal, loss_anomaly)
        return loss.mean()
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
def estimate_cov_streaming(net, loader_norm, c, device):
    net.eval(); sum_x, sum_xx, N = None, None, 0
    for (xb,) in loader_norm:
        xb = xb.to(device)
        z = net(xb)
        diff = (z - c).detach().cpu().numpy().astype(np.float64)
        B, d = diff.shape
        if sum_x is None: sum_x, sum_xx = np.zeros(d, dtype=np.float64), np.zeros((d, d), dtype=np.float64)
        sum_x += diff.sum(axis=0)
        sum_xx += diff.T @ diff
        N += B
    mu = sum_x / N; S = (sum_xx / N) - np.outer(mu, mu); return (S + S.T) * 0.5
def shrink_inv(S, alpha):
    d = S.shape[0]; tr_over_d = np.trace(S) / d
    S_shrunk = (1.0 - alpha) * S + alpha * (tr_over_d) * np.eye(d, dtype=S.dtype)
    try: return np.linalg.inv(S_shrunk).astype(np.float32)
    except np.linalg.LinAlgError: return np.linalg.pinv(S_shrunk).astype(np.float32)
@torch.no_grad()
def svdd_score_mahalanobis_loader(net, loader, c, Pinv, device):
    net.eval(); P = torch.as_tensor(Pinv, dtype=torch.float32, device=device); scores = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device)
        z = net(xb)
        diff = z - c
        m = torch.einsum('bi,ij,bj->b', diff, P, diff)
        scores.append(m.detach().cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=np.float32)

def get_svdd_scores(Xtr, ytr, Xval, yval, Xte, rk_val, rk_test, scaler, original_Xtr_df_for_scaler, original_ytr_for_scaler):
    print("\n" + "="*80)
    print(">>> FASE 1: Treinando Modelo Base DEEP SVDD <<<")
    print("="*80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dl_train = DataLoader(NPDataset(Xtr, ytr), batch_size=SVDD_BATCH_TRAIN, shuffle=True)
    mask_norm_tr_orig = (original_ytr_for_scaler == 0)
    Xtr_norm_only = scaler.transform(original_Xtr_df_for_scaler[mask_norm_tr_orig])
    dl_norm_train = DataLoader(NPDataset(Xtr_norm_only), batch_size=SVDD_BATCH_EVAL, shuffle=False)
    dl_val  = DataLoader(NPDataset(Xval, yval), batch_size=SVDD_BATCH_EVAL, shuffle=False)
    dl_test = DataLoader(NPDataset(Xte), batch_size=SVDD_BATCH_EVAL, shuffle=False)

    net = DeepSVDDNet(d_in=Xtr.shape[1]).to(device)
    c = init_center_c(net, dl_norm_train, device)
    
    # Treinamento
    net.train()
    with torch.no_grad():
        all_dists_sq = []
        for (xb,) in dl_norm_train:
            xb = xb.to(device)
            dist_sq = torch.sum((net(xb) - c)**2, dim=1)
            all_dists_sq.append(dist_sq.cpu().numpy())
    R_squared = np.quantile(np.concatenate(all_dists_sq), SVDD_R_QUANTILE)
    criterion = SVDDLoss(c, R_squared).to(device)
    optimizer = optim.Adam(net.parameters(), lr=SVDD_LR, weight_decay=SVDD_WD)
    for ep in range(1, SVDD_EPOCHS + 1):
        t0, ep_loss = time.time(), 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward(); optimizer.step()
            ep_loss += loss.item()
        print(f"[SVDD] Treino Ep {ep:02d}/{SVDD_EPOCHS} | Loss={ep_loss/len(dl_train):.6f} | {time.time()-t0:.1f}s")

    # Geração de scores
    S = estimate_cov_streaming(net, dl_norm_train, c, device)
    S_inv = shrink_inv(S, SVDD_SHRINKAGE)
    
    scores_val_raw = svdd_score_mahalanobis_loader(net, dl_val, c, S_inv, device)
    scores_test_raw = svdd_score_mahalanobis_loader(net, dl_test, c, S_inv, device)
    
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(scores_val_raw, yval, rk_val)
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    scores_test_g = apply_group_zscore(scores_test_raw, rk_test, mu_map, sd_map, g_mean, g_std)
    
    print("[SVDD] Scores de validação e teste gerados com sucesso.")
    return scores_val_g, scores_test_g

# ================================================================
# SEÇÃO 2: Definições e Lógica do Modelo LightGBM
# ================================================================
def get_lgbm_scores(Xtr, ytr, Xval, yval, Xte, rk_val, rk_test):
    print("\n" + "="*80)
    print(">>> FASE 2: Treinando Modelo Base LIGHTGBM <<<")
    print("="*80)
    
    neg_count = np.sum(ytr == 0); pos_count = np.sum(ytr == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    params = LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, ytr,
              eval_set=[(Xval, yval)], eval_metric='f1',
              callbacks=[lgb.early_stopping(150, verbose=True)])

    scores_val_raw = model.predict_proba(Xval)[:, 1]
    scores_test_raw = model.predict_proba(Xte)[:, 1]
    
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(scores_val_raw, yval, rk_val)
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    scores_test_g = apply_group_zscore(scores_test_raw, rk_test, mu_map, sd_map, g_mean, g_std)
    
    print("[LGBM] Scores de validação e teste gerados com sucesso.")
    return scores_val_g, scores_test_g

# ================================================================
# SEÇÃO 3: Fluxo Principal de Stacking
# ================================================================
if __name__ == "__main__":
    set_seed(SEED)
    
    # --- PASSO 1: Carregar e preparar todos os dados ---
    print("[MAIN] Iniciando pipeline de dados...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS)
    X_all_df = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    X_te_df  = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)
    rk_test = make_route_keys(test_df)

    # --- PASSO 2: Split, SMOTE e Scaling ---
    print("[MAIN] Realizando split, SMOTE e scaling...")
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)
    
    original_Xtr_df_for_scaler, _, original_ytr_for_scaler, _ = train_test_split(
        X_all_df, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)

    if USE_SMOTE:
        print("[MAIN] Aplicando SMOTE no conjunto de treino...")
        smote = SMOTE(random_state=SEED)
        Xtr_df_resampled, ytr_resampled = smote.fit_resample(Xtr_df, ytr)
    else:
        Xtr_df_resampled, ytr_resampled = Xtr_df, ytr

    mask_norm_tr_orig = (original_ytr_for_scaler == 0)
    scaler = StandardScaler().fit(original_Xtr_df_for_scaler[mask_norm_tr_orig])
    Xtr_s = scaler.transform(Xtr_df_resampled)
    Xval_s = scaler.transform(Xval_df)
    Xte_s = scaler.transform(X_te_df)

    # --- PASSO 3: Treinar modelos base e obter scores de validação/teste ---
    scores_val_svdd, scores_test_svdd = get_svdd_scores(
        Xtr_s, ytr_resampled, Xval_s, yval, Xte_s, rk_val, rk_test, scaler, 
        original_Xtr_df_for_scaler, original_ytr_for_scaler)

    scores_val_lgbm, scores_test_lgbm = get_lgbm_scores(
        Xtr_s, ytr_resampled, Xval_s, yval, Xte_s, rk_val, rk_test)

    # --- PASSO 4: Treinar o meta-modelo ---
    print("\n" + "="*80)
    print(">>> FASE 3: Treinando Meta-Modelo de Stacking <<<")
    print("="*80)
    
    X_meta_train = np.stack([scores_val_svdd, scores_val_lgbm], axis=1)
    y_meta_train = yval
    
    meta_model = LogisticRegression(class_weight='balanced', random_state=SEED, C=0.1, solver='liblinear')
    meta_model.fit(X_meta_train, y_meta_train)
    print("[META] Meta-modelo treinado.")

    # --- PASSO 5: Avaliar o ensemble completo na validação ---
    scores_meta_val = meta_model.predict_proba(X_meta_train)[:, 1]
    best_thr_meta, _ = calibrate_threshold_by_validation(scores_meta_val, y_meta_train)
    ypred_meta_val = (scores_meta_val > best_thr_meta).astype(int)
    
    print("\n[RESULTADO FINAL] Classification report (val, STACKING ENSEMBLE):")
    print(classification_report(y_meta_train, ypred_meta_val, digits=4))

    # --- PASSO 6: Gerar a submissão final ---
    print("\n[MAIN] Gerando predições finais no conjunto de teste...")
    X_meta_test = np.stack([scores_test_svdd, scores_test_lgbm], axis=1)
    final_scores_test = meta_model.predict_proba(X_meta_test)[:, 1]
    ypred_test_final = (final_scores_test > best_thr_meta).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test_final, return_counts=True)
    print("[MAIN] Distribuição das predições finais (teste):")
    for u, c_ in zip(uniq_pred, cnt_pred): print(f"[MAIN]   classe {u}: {c_}")

    print("\n[MAIN] Salvando submission_stacking.csv...")
    sub = pd.DataFrame({"id": test_df["tr_id"].astype(int), "target": ypred_test_final})
    sub.to_csv("submission_stacking.csv", index=False)
    print(f"[MAIN] Arquivo salvo: {os.path.abspath('submission_stacking.csv')}")
    print("\n[MAIN] Processo concluído com sucesso!")
