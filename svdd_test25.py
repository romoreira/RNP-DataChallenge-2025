# lgbm_hyperopt_advanced.py
# Script para otimização de hiperparâmetros com Hyperopt,
# utilizando o novo conjunto de features ADVANCED.

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Imports para Hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# =======================
# Configurações principais
# =======================
SEED = 42
VAL_SIZE = 0.20

# ========== ALTERAÇÕES AQUI ==========
# Apontar para o novo diretório e tag de cache das features avançadas
CACHE_DIR   = "./_cache_feats_ADVANCED"
CACHE_TAG   = "ADVANCED"
# =====================================

USE_CACHE   = True
CSV_ENGINE  = "pyarrow"
TRAIN_CSV   = "dataset/train.csv"
TEST_CSV    = "dataset/test.csv"
NROWS       = None

# Normalização por grupo
MIN_GROUP_NORMALS = 100
EPS_STD           = 1e-8

# ================================================================
# Funções de utilidade, featurização e dataset
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
    
# Esta função não será usada se o cache for encontrado, mas é mantida aqui.
def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    print("[WARN] A função featurize_fast não foi definida para as features ADVANCED. O cache DEVE existir.")
    return pd.DataFrame() 

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if USE_CACHE and os.path.exists(cache_path):
        try:
            t0 = time.time()
            df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) >= (nrows or 0)):
                 print(f"[INFO] Carregado do cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
                 if nrows is not None:
                     return df.head(nrows)
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
    best_thr = float(s[j]) if len(s) > 0 else 0.5
    return best_thr

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # ----- Leitura e Featurização -----
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    
    X_all_df = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)

    # ----- Split estratificado -----
    print(f"[INFO] Split estratificado train/val (val_size={VAL_SIZE}) ...")
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    print(f"[INFO] Xtrain: {Xtr_df.shape}, Xval: {Xval_df.shape}")

    # ----- Scaler -----
    print("[INFO] Ajustando StandardScaler SOMENTE nos normais do treino...")
    mask_norm_tr_orig = (ytr == 0)
    scaler = StandardScaler().fit(Xtr_df[mask_norm_tr_orig])
    
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)

    # =================================================================
    #  >>> Bloco de Otimização com Hyperopt <<<
    # =================================================================
    
    # 1. Definir o espaço de busca de hiperparâmetros
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.1)),
        'num_leaves': hp.quniform('num_leaves', 20, 60, 1),
        'max_depth': hp.quniform('max_depth', 5, 12, 1),
        'scale_pos_weight': hp.quniform('scale_pos_weight', 5, 25, 1),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.1), np.log(20.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(20.0)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
    }

    # 2. Definir a função objetivo a ser minimizada
    def objective(params):
        """Função que treina e avalia o modelo para um conjunto de params."""
        
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        
        params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': 2500,
            'seed': SEED,
            'n_jobs': -1,
            'verbose': -1,
        })
        
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr,
                  eval_set=[(Xval, yval)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        scores_val_raw = model.predict_proba(Xval)[:, 1]
        
        mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(
            scores_val_raw, yval, rk_val, min_group_normals=MIN_GROUP_NORMALS
        )
        
        scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
        
        best_thr = calibrate_threshold_by_validation(scores_val_g, yval)
        ypred_val = (scores_val_g > best_thr).astype(int)
        
        f1_macro = f1_score(yval, ypred_val, average='macro')
        
        loss = 1 - f1_macro
        
        print(f"F1-Macro: {f1_macro:.4f} | Loss: {loss:.4f} | Params: nl={params['num_leaves']}, spw={params['scale_pos_weight']:.1f}, rl={params['reg_lambda']:.2f}")
        
        return {'loss': loss, 'status': STATUS_OK}

    # 3. Executar a otimização
    print("\n[INFO] --- Iniciando Otimização de Hiperparâmetros com Hyperopt para features ADVANCED ---")
    
    trials = Trials()
    max_evaluations = 50 

    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=trials
    )
    
    print("\n[INFO] --- Otimização Concluída ---")
    print("\n[INFO] Melhores parâmetros encontrados (valores brutos):")
    print(best_params)
    
    best_params_clean = best_params.copy()
    best_params_clean['num_leaves'] = int(trials.best_trial['misc']['vals']['num_leaves'][0])
    best_params_clean['max_depth'] = int(trials.best_trial['misc']['vals']['max_depth'][0])
    best_params_clean['scale_pos_weight'] = int(trials.best_trial['misc']['vals']['scale_pos_weight'][0])
    
    print("\n[INFO] Dicionário de parâmetros pronto para uso:")
    print(best_params_clean)
