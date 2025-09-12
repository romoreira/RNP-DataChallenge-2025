# lgbm_final_rolling.py
# Script final para treinar o modelo LGBM com os melhores hiperparâmetros
# e gerar a submissão, utilizando o conjunto de features ROLLING1.

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# =======================
# Configurações principais
# =======================
SEED = 42
VAL_SIZE = 0.20

# ========== Parâmetros do modelo (extraídos da otimização) ==========
BEST_PARAMS = {
    'colsample_bytree': 0.6461385544750221,
    'learning_rate': 0.09284053521608261,
    'max_depth': 13,
    'num_leaves': 52,
    'reg_alpha': 7.489328711179947,
    'reg_lambda': 8.5793439036779,
    'scale_pos_weight': 3,
    'subsample': 0.6327933187152609,
}
# ====================================================================

# ========== ALTERAÇÕES AQUI ==========
CACHE_DIR = "./_cache_feats_FINAL"
CACHE_TAG = "FINAL1"
# =====================================

TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
CSV_ENGINE = "pyarrow"
NROWS = None

# Normalização por grupo
MIN_GROUP_NORMALS = 100
EPS_STD = 1e-8

# ================================================================
# Funções de utilidade e pós-processamento
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

def load_features_from_cache(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Arquivo de cache não encontrado: {cache_path}. Execute o script 'generate_advanced_features.py' primeiro.")
    t0 = time.time()
    df = pd.read_parquet(cache_path)
    print(f"[INFO] Carregado do cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

def make_route_keys(df: pd.DataFrame) -> np.ndarray:
    return (df["tr_src"].astype(np.int64).values << 32) ^ df["tr_dst"].astype(np.int64).values

def compute_group_norm_stats(scores: np.ndarray, y: np.ndarray, keys: np.ndarray,
                            min_group_normals=100, eps_std=1e-8):
    mask_norm = (y == 0)
    s_norm, k_norm = scores[mask_norm], keys[mask_norm]
    g_mean = float(s_norm.mean()) if len(s_norm) > 0 else 0.5
    g_std = float(s_norm.std() + eps_std) if len(s_norm) > 0 else 1.0
    df = pd.DataFrame({"k": k_norm, "s": s_norm})
    grp = df.groupby("k")["s"]
    mu, sd, cnt = grp.mean(), grp.std().fillna(0.0), grp.size()
    mu_adj = mu.where(cnt >= min_group_normals, g_mean)
    sd_adj = sd.where((cnt >= min_group_normals) & (sd > 0.0), g_std)
    mu_map, sd_map = mu_adj.to_dict(), sd_adj.to_dict()
    print(f"[INFO] Grupos na VAL: {len(mu_map)} | com >= {min_group_normals} normais: {int((cnt >= min_group_normals).sum())} | g_std={g_std:.6f}")
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
    total_pos = int(cum_pos[-1]) if len(cum_pos) > 0 else 0
    TP = total_pos - cum_pos
    FP = (len(y) - total_pos) - cum_neg
    FN = cum_pos
    prec1 = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) > 0)
    rec1 = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) > 0)
    f1_1 = np.divide(2 * prec1 * rec1, (prec1 + rec1), out=np.zeros_like(prec1), where=(prec1 + rec1) > 0)
    if not len(f1_1): return 0.5
    j = int(np.argmax(f1_1))
    best_thr = float(s[j]) if len(s) > 0 else 0.5
    print(f"[INFO] Calibração (otimizando F1-score classe 1): melhor F1-1={f1_1[j]:.6f} @ thr={best_thr:.6e}")
    return best_thr

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # 1. Carregar os dados de treino e teste
    print("\n[INFO] Carregando dados de treino...")
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS)
    test_df = read_csv_fast(TEST_CSV, nrows=NROWS)
    
    # 2. Carregar as features
    print("[INFO] Carregando features do cache...")
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    X_all_df = load_features_from_cache(cache_train)
    X_te_df = load_features_from_cache(cache_test)
    
    # 3. Preparar os dados para treinamento e validação
    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)
    rk_test = make_route_keys(test_df)
    
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    
    # 4. Ajustar e aplicar o scaler
    print("[INFO] Ajustando StandardScaler...")
    scaler = StandardScaler().fit(Xtr_df[ytr == 0])
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    Xtest = scaler.transform(X_te_df)

    # 5. Treinar o modelo LightGBM com os melhores hiperparâmetros
    print("\n[INFO] Treinando o modelo LightGBM com os melhores parâmetros...")
    lgbm_model = lgb.LGBMClassifier(**BEST_PARAMS,
                                    objective='binary',
                                    metric='auc',
                                    boosting_type='gbdt',
                                    n_estimators=3000,
                                    seed=SEED,
                                    n_jobs=-1,
                                    verbose=-1)
    
    lgbm_model.fit(Xtr, ytr,
                   eval_set=[(Xval, yval)],
                   eval_metric='auc',
                   callbacks=[lgb.early_stopping(150, verbose=False)])

    # 6. Pós-processamento e Calibração
    print("[INFO] Pós-processamento e calibração...")
    scores_val_raw = lgbm_model.predict_proba(Xval)[:, 1]
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(scores_val_raw, yval, rk_val)
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    best_thr = calibrate_threshold_by_validation(scores_val_g, yval)
    
    ypred_val = (scores_val_g > best_thr).astype(int)
    print("[INFO] Performance na validação:")
    print(classification_report(yval, ypred_val, digits=4))

    # 7. Gerar predições e submissão
    print("\n[INFO] Gerando predições no conjunto de teste...")
    scores_test_raw = lgbm_model.predict_proba(Xtest)[:, 1]
    scores_test_g = apply_group_zscore(scores_test_raw, rk_test, mu_map, sd_map, g_mean, g_std)
    ypred_test = (scores_test_g > best_thr).astype(int)

    uniq_pred, cnt_pred = np.unique(ypred_test, return_counts=True)
    print("[INFO] Distribuição das predições (teste):")
    for u, c in zip(uniq_pred, cnt_pred):
        print(f"[INFO]   classe {u}: {c}")

    print("[INFO] Salvando submission.csv...")
    submission_df = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_test.astype(int)
    })
    submission_df.to_csv("submission.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission.csv')}")
