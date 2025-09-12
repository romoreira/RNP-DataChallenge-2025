# lgbm_test01.py
# Modelo LightGBM com pipeline de features DELTA1 e pós-processamento
# de normalização de score por grupo para maximizar F1-score.

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

# =======================
# Configurações principais
# =======================
SEED = 42
VAL_SIZE = 0.20
USE_SMOTE = False # Mantenha True para consistência com o teste anterior

CACHE_DIR   = "./_cache_feats"
CACHE_TAG   = "DELTA1"
USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# Normalização por grupo
MIN_GROUP_NORMALS = 100
EPS_STD           = 1e-8

# ================================================================
# Funções de utilidade, featurização e dataset (SEM ALTERAÇÕES)
# ================================================================
def set_seed(seed=42):
    np.random.seed(seed)

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
    if not isinstance(s, str) or len(s) < 2:
        return np.array([], dtype=np.float32)
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
    out["success_rate"]         = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"]            = 1.0 - out["success_rate"]
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

# ================================================================
# Funções de Normalização e Calibração (SEM ALTERAÇÕES)
# ================================================================
def make_route_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["tr_src"].astype(np.int64).values << 32) ^ df["tr_dst"].astype(np.int64).values

def compute_group_norm_stats(scores: np.ndarray, y: np.ndarray, keys: np.ndarray,
                             min_group_normals=100, eps_std=1e-8):
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
    print(f"[INFO] Calibração exata (otimizando F1-score da classe 1): melhor F1-1={best:.6f} @ thr={best_thr:.6e}")
    return best_thr, best

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # ----- Leitura e Featurização (com cache)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS, use_engine=CSV_ENGINE)
    
    X_all_df = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    X_te_df  = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)

    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)
    rk_test = make_route_keys(test_df)

    # ----- Split estratificado
    print(f"[INFO] Split estratificado train/val (val_size={VAL_SIZE}) ...")
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"[INFO] Xtrain: {Xtr_df.shape}, Xval: {Xval_df.shape}")

    # ----- SMOTE (Opcional)
    if USE_SMOTE:
        print("[INFO] Aplicando SMOTE no conjunto de treino...")
        print(f"[INFO] Contagem antes: 0={np.sum(ytr==0)}, 1={np.sum(ytr==1)}")
        smote = SMOTE(random_state=SEED)
        Xtr_df, ytr = smote.fit_resample(Xtr_df, ytr)
        print(f"[INFO] Contagem depois: 0={np.sum(ytr==0)}, 1={np.sum(ytr==1)}")
    
    # ----- Scaler (ajustado SOMENTE nos normais do treino original)
    print("[INFO] Ajustando StandardScaler SOMENTE nos normais do treino...")
    original_Xtr_df, _, original_ytr, _ = train_test_split(X_all_df, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)
    mask_norm_tr_orig = (original_ytr == 0)
    scaler = StandardScaler().fit(original_Xtr_df[mask_norm_tr_orig])
    
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    Xte = scaler.transform(X_te_df)

    # =================================================================
    #  >>> Bloco Principal: Modelo LightGBM <<<
    # =================================================================
    print("\n[INFO] --- Iniciando Treinamento e Avaliação com LightGBM ---")
    
    # 1. Calcular peso das classes
    neg_count = np.sum(ytr == 0)
    pos_count = np.sum(ytr == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    scale_pos_weight = 10
    print(f"[INFO] LightGBM scale_pos_weight = {scale_pos_weight:.2f}")

    # 2. Parâmetros do LightGBM
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'n_estimators': 2500, 'learning_rate': 0.01, 'num_leaves': 40,
        'max_depth': 10, 'seed': SEED, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
        'reg_lambda': 0.1, 'scale_pos_weight': scale_pos_weight,
    }
    
    # 3. Treinar modelo com early stopping
    print("[INFO] Treinando o modelo LightGBM...")
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, ytr,
              eval_set=[(Xval, yval)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(150, verbose=True)])
              
    # ----- Avaliação com Normalização por Grupo
    print("\n[INFO] Computando scores (probabilidades) na VAL...")
    scores_val_raw = model.predict_proba(Xval)[:, 1]
    
    print("[INFO] Estimando estatísticas por grupo na VAL (somente normais)...")
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(
        scores_val_raw, yval, rk_val,
        min_group_normals=MIN_GROUP_NORMALS, eps_std=EPS_STD
    )
    # Aplica a normalização Z-score nas probabilidades
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    
    # ----- Calibração e Validação
    # Calibra o limiar nos scores normalizados
    best_thr, best_f1_val = calibrate_threshold_by_validation(scores_val_g, yval)
    ypred_val = (scores_val_g > best_thr).astype(int)

    print("\n[INFO] Classification report (val, LightGBM):")
    print(classification_report(yval, ypred_val, digits=4))

    # ----- Geração de Submissão
    print("\n[INFO] Gerando predições no TESTE (com normalização por grupo)...")
    scores_test_raw = model.predict_proba(Xte)[:, 1]
    scores_test_g   = apply_group_zscore(scores_test_raw, rk_test, mu_map, sd_map, g_mean, g_std)
    ypred_test = (scores_test_g > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[INFO] Distribuição das predições (teste):")
    for u, c_ in zip(uniq_pred, cnt_pred): print(f"[INFO]   classe {u}: {c_}")

    print("\n[INFO] Salvando submission_lightgbm.csv...")
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    sub.to_csv("submission_lightgbm.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission_lightgbm.csv')}")
