# lgbm_optimized_final.py
# Versão final e otimizada utilizando APENAS o modelo LightGBM.
# Este script consolida:
# 1. A melhor engenharia de features (com janelas deslizantes/rolling).
# 2. Os melhores hiperparâmetros encontrados via otimização (Hyperopt).
# 3. A pipeline completa de pós-processamento (normalização por grupo e calibração).

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =======================
# Configurações Globais
# =======================
SEED = 42
VAL_SIZE = 0.20

# --- Configs de IO e Cache ---
CACHE_DIR   = "./_cache_feats"
CACHE_TAG   = "FINAL_WITH_ROLLING" # Usando o cache com as features mais recentes
USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# --- Configs de Pós-processamento ---
MIN_GROUP_NORMALS = 100
EPS_STD           = 1e-8

# --- Configs do Modelo LGBM (OTIMIZADAS PELO HYPEROPT) ---
LGBM_BEST_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 3000, # Usamos um número alto, o early stopping encontrará o ideal
    'learning_rate': 0.01,
    'seed': SEED,
    'n_jobs': -1,
    'verbose': -1,
    # --- PARÂMETROS ENCONTRADOS PELA OTIMIZAÇÃO ---
    'num_leaves': 52,
    'scale_pos_weight': 3.0,
    'reg_lambda': 8.58,
    # --- Parâmetros de regularização adicionais para robustez ---
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
}

# ================================================================
# Funções de Preparação e Pós-Processamento
# (As funções são as mesmas da versão de Stacking, otimizadas e consolidadas)
# ================================================================

def set_seed(seed=42):
    np.random.seed(seed)

def read_csv_fast(path, nrows=None, use_engine="pyarrow"):
    t0 = time.time()
    kw = dict(nrows=nrows)
    try:
        df = pd.read_csv(path, engine="pyarrow", **kw)
        print(f"[INFO] read_csv (pyarrow) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
        return df
    except Exception as e:
        print(f"[WARN] pyarrow falhou ({e}). Caindo para engine='c'.")
    df = pd.read_csv(path, engine="c", **kw)
    print(f"[INFO] read_csv (c) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

def featurize_final_with_rolling(df: pd.DataFrame) -> pd.DataFrame:
    print("[Featurize] Iniciando featurização base (DELTA)...")
    t0 = time.time()
    rtts = df["all_rtts"].apply(lambda s: np.fromstring(s[1:-1], sep=',') if isinstance(s, str) and len(s) > 1 else np.array([], dtype=np.float32))
    n = len(rtts)
    out = pd.DataFrame({
        "rtt_mean": np.fromiter((a.mean() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n),
        "rtt_std": np.fromiter((a.std(ddof=0) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n),
        "rtt_p90": np.fromiter((np.quantile(a, 0.9) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n),
        "total_probes_sent": df["total_probes_sent"].astype(np.float32),
        "total_replies_last_hop": df["total_replies_last_hop"].astype(np.float32),
        "seconds_since_start": df["seconds_since_start"].astype(np.float32),
    })
    eps = 1e-9
    out["success_rate"] = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    required = {"tr_src","tr_dst","seconds_since_start"}
    if required.issubset(df.columns):
        tmp = out.copy()
        tmp["tr_src"] = df["tr_src"]
        tmp["tr_dst"] = df["tr_dst"]
        tmp.sort_values(["tr_src","tr_dst","seconds_since_start"], kind="mergesort", inplace=True)

        for base_col in ["rtt_mean", "rtt_p90", "success_rate"]:
            prev = tmp.groupby(["tr_src","tr_dst"])[base_col].shift(1)
            out[f"delta_{base_col}"] = (tmp[base_col] - prev).astype(np.float32).values
        out["time_since_prev"] = tmp.groupby(["tr_src","tr_dst"])["seconds_since_start"].diff().fillna(0.0).astype(np.float32).values
        out["is_first_obs"] = (tmp.groupby(["tr_src","tr_dst"]).cumcount() == 0).astype(np.float32).values
        
        print("[Featurize] Adicionando features de janela deslizante (ROLLING)...")
        cols_to_roll = ['rtt_mean', 'rtt_std', 'rtt_p90', 'success_rate']
        windows = [3, 5, 7]
        for col in cols_to_roll:
            for w in windows:
                rolled = tmp.groupby(['tr_src', 'tr_dst'])[col].rolling(window=w, min_periods=1)
                out[f'{col}_roll{w}_mean'] = rolled.mean().reset_index(level=[0,1], drop=True).astype(np.float32).values
                out[f'{col}_roll{w}_std'] = rolled.std().fillna(0).reset_index(level=[0,1], drop=True).astype(np.float32).values

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    print(f"[Featurize] Featurização final concluída: {out.shape} | {time.time()-t0:.1f}s")
    return out

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if USE_CACHE and os.path.exists(cache_path):
        try:
            t0 = time.time(); df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) == nrows):
                print(f"[INFO] Carregado do cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
                return df
        except Exception as e: print(f"[WARN] Falha ao ler cache ({e}). Recriando.")
    base = read_csv_fast(csv_path, nrows=nrows)
    feats = featurize_final_with_rolling(base)
    if USE_CACHE:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats.to_parquet(cache_path, index=False)
        print(f"[INFO] Cache salvo em: {cache_path}")
    return feats

def make_route_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["tr_src"].astype(np.int64).values << 32) ^ df["tr_dst"].astype(np.int64).values

def compute_group_norm_stats(scores, y, keys, min_group_normals=MIN_GROUP_NORMALS, eps_std=EPS_STD):
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
    print(f"[INFO] Grupos na VAL: {len(mu_map)} | com >= {min_group_normals} normais: {int((cnt >= min_group_normals).sum())}")
    return mu_map, sd_map, g_mean, g_std

def apply_group_zscore(scores, keys, mu_map, sd_map, g_mean, g_std):
    ks = pd.Series(keys)
    mu = ks.map(mu_map).astype(float).fillna(g_mean).values
    sd = ks.map(sd_map).astype(float).fillna(g_std).values
    sd = np.where(sd <= 0.0, g_std, sd)
    return ((scores - mu) / sd).astype(np.float32)

def calibrate_threshold_by_validation(scores_val, yval):
    idx = np.argsort(scores_val)
    s, y = scores_val[idx].astype(np.float64), yval[idx].astype(int)
    pos = (y == 1)
    TP = np.sum(pos) - np.cumsum(pos)
    FP = (len(y) - np.sum(pos)) - np.cumsum(~pos)
    FN = np.cumsum(pos)
    prec1 = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) > 0)
    rec1 = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) > 0)
    f1_1 = np.divide(2 * prec1 * rec1, prec1 + rec1, out=np.zeros_like(prec1), where=(prec1 + rec1) > 0)
    j = np.argmax(f1_1)
    best_thr, best_f1 = float(s[j]), float(f1_1[j])
    print(f"[INFO] Calibração (F1-score classe 1): melhor F1-1={best_f1:.6f} @ thr={best_thr:.6e}")
    return best_thr, best_f1

# ================================================================
# Fluxo Principal (Single Model Otimizado)
# ================================================================
if __name__ == "__main__":
    set_seed(SEED)
    
    # --- PASSO 1: Carregar e preparar dados com features avançadas ---
    print("[MAIN] Iniciando pipeline de dados...")
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    X_all_df = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    X_te_df  = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)
    rk_test = make_route_keys(test_df)

    # --- PASSO 2: Split e Scaling ---
    print("[MAIN] Realizando split e scaling...")
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)
    
    mask_norm_tr = (ytr == 0)
    scaler = StandardScaler().fit(Xtr_df[mask_norm_tr])
    Xtr_s = scaler.transform(Xtr_df)
    Xval_s = scaler.transform(Xval_df)
    Xte_s = scaler.transform(X_te_df)

    # --- PASSO 3: Treinar o Modelo LightGBM Otimizado ---
    print("\n" + "="*80 + "\n>>> Treinando Modelo Final LIGHTGBM (Otimizado) <<<\n" + "="*80)
    
    model = lgb.LGBMClassifier(**LGBM_BEST_PARAMS)
    model.fit(Xtr_s, ytr,
              eval_set=[(Xval_s, yval)],
              eval_metric='f1',
              callbacks=[lgb.early_stopping(150, verbose=True)])

    # --- PASSO 4: Avaliação na Validação com Pós-Processamento ---
    print("\n[MAIN] Avaliando modelo no conjunto de validação...")
    scores_val_raw = model.predict_proba(Xval_s)[:, 1]
    
    print("[MAIN] Aplicando normalização por grupo nos scores de validação...")
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(scores_val_raw, yval, rk_val)
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    
    print("[MAIN] Calibrando limiar final nos scores normalizados...")
    best_thr, best_f1 = calibrate_threshold_by_validation(scores_val_g, yval)
    ypred_val = (scores_val_g > best_thr).astype(int)
    
    print("\n[RESULTADO FINAL] Classification report (validação):")
    print(classification_report(yval, ypred_val, digits=4))

    # --- PASSO 5: Geração da Submissão Final ---
    print("\n[MAIN] Gerando predições finais no conjunto de teste...")
    scores_test_raw = model.predict_proba(Xte_s)[:, 1]
    scores_test_g   = apply_group_zscore(scores_test_raw, rk_test, mu_map, sd_map, g_mean, g_std)
    ypred_test = (scores_test_g > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[MAIN] Distribuição das predições finais (teste):")
    for u, c in zip(uniq_pred, cnt_pred): print(f"[MAIN]   classe {u}: {c}")

    print("\n[MAIN] Salvando submission_lgbm_optimized.csv...")
    sub = pd.DataFrame({"id": test_df["tr_id"].astype(int), "target": ypred_test})
    sub.to_csv("submission_lgbm_optimized.csv", index=False)
    print(f"[MAIN] Arquivo salvo: {os.path.abspath('submission_lgbm_optimized.csv')}")
    print("\n[MAIN] Processo concluído com sucesso!")
