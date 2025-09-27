# svdd_test30_stacking.py
# Script com features categóricas para CatBoost e stacking ensemble.

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression # NEW: Import for Stacking
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# =======================
# Configurações principais
# =======================
SEED = 42
N_SPLITS = 5
USE_CACHE = True
CACHE_DIR = "./_cache_feats"
CACHE_TAG = "DELTA4_CAT_FEATS" # NEW: New tag because the feature set changed
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
NROWS = None

# ================================================================
# Funções de utilidade e featurização (COM ALTERAÇÕES)
# ================================================================
def set_seed(seed=42):
    np.random.seed(seed)

def read_csv_fast(path, nrows=None, use_engine="pyarrow"):
    # (This function remains unchanged)
    t0 = time.time()
    kw = dict(nrows=nrows)
    try:
        df = pd.read_csv(path, engine="pyarrow", **kw)
    except Exception:
        df = pd.read_csv(path, engine="c", **kw)
    print(f"[INFO] read_csv OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    # (This function is modified to keep tr_src and tr_dst)
    t0 = time.time()
    rtts = df["all_rtts"].apply(lambda s: np.fromstring(s[1:-1], sep=',', dtype=np.float32) if isinstance(s, str) and len(s) > 2 else np.array([], dtype=np.float32))
    stats = [
        (np.mean(a) if a.size else 0.0, np.std(a, ddof=0) if a.size else 0.0,
         np.percentile(a, [25, 50, 75, 90]) if a.size else [0.0]*4,
         np.min(a) if a.size else 0.0, np.max(a) if a.size else 0.0, a.size)
        for a in rtts
    ]
    means, stds, percentiles, mins, maxs, lens = zip(*stats)
    percentiles = np.array(percentiles)
    
    out = pd.DataFrame({
        "rtt_mean": means, "rtt_std": stds, "rtt_p25": percentiles[:, 0],
        "rtt_median": percentiles[:, 1], "rtt_p75": percentiles[:, 2],
        "rtt_p90": percentiles[:, 3], "rtt_min": mins, "rtt_max": maxs, "rtt_len": lens,
        "tr_attempts": df["tr_attempts"], "total_probes_sent": df["total_probes_sent"],
        "total_replies_last_hop": df["total_replies_last_hop"],
        "seconds_since_start": df["seconds_since_start"],
    })
    out['rtt_iqr'] = out['rtt_p75'] - out['rtt_p25']
    eps = 1e-9
    out["success_rate"] = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"] = 1.0 - out["success_rate"]
    
    tmp = out.copy()
    tmp["tr_src"] = df["tr_src"]
    tmp["tr_dst"] = df["tr_dst"]
    tmp.sort_values(["tr_src", "tr_dst", "seconds_since_start"], inplace=True)
    
    grouped = tmp.groupby(["tr_src", "tr_dst"])
    base_cols_for_delta = ["rtt_mean", "rtt_std", "rtt_p90", "success_rate"]
    for col in base_cols_for_delta:
        tmp[f'delta_{col}'] = grouped[col].diff()
        ratio = tmp[col] / grouped[col].shift().replace(0, np.nan)
        tmp[f'ratio_{col}'] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 10.0)
    tmp["time_since_prev"] = grouped["seconds_since_start"].diff().fillna(0.0)
    
    base_cols_for_rolling = ['rtt_mean', 'rtt_std', 'success_rate']
    window_sizes = [3, 7]
    for col in base_cols_for_rolling:
        for w in window_sizes:
            rolling_grp = grouped[col].rolling(window=w, min_periods=1)
            tmp[f'rolling_mean_{col}_w{w}'] = rolling_grp.mean().reset_index(level=[0,1], drop=True)
            tmp[f'rolling_std_{col}_w{w}'] = rolling_grp.std().reset_index(level=[0,1], drop=True).fillna(0)

    # MODIFIED: Instead of dropping tr_src and tr_dst, we just sort back to the original index
    out = tmp.sort_index()
    out = out.fillna(0.0).astype(np.float32, errors='ignore') # ignore errors for object types
    print(f"[INFO] featurize_fast: {out.shape} | {time.time()-t0:.1f}s")
    return out

def maybe_cached_features(csv_path, cache_path, nrows=None):
    # (This function remains unchanged)
    if USE_CACHE and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) == nrows):
                print(f"[INFO] Carregado do cache: {cache_path} | {df.shape}")
                return df
        except Exception: pass
    base = read_csv_fast(csv_path, nrows=nrows)
    feats = featurize_fast(base)
    if USE_CACHE:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats.to_parquet(cache_path, index=False)
    return feats

def calibrate_threshold_by_validation(scores_val, yval):
    # (This function remains unchanged)
    best_thr, best_f1 = 0.5, -1.0
    thresholds = np.linspace(scores_val.min(), scores_val.max(), 500)
    for thr in thresholds:
        preds = (scores_val > thr).astype(int)
        f1 = f1_score(yval, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"[INFO] Melhor limiar F1 encontrado: {best_thr:.4f} (F1: {best_f1:.4f})")
    return best_thr

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # --- Leitura e Featurização ---
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS)
    test_df = read_csv_fast(TEST_CSV, nrows=NROWS)
    X_all_df = maybe_cached_features(TRAIN_CSV, os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}.parquet"), nrows=NROWS)
    X_test_df = maybe_cached_features(TEST_CSV, os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}.parquet"), nrows=NROWS)
    y_all = train_df["route_changed"].astype(int).values

    # NEW: Identify categorical features for CatBoost
    feature_names = list(X_all_df.columns)
    categorical_features_indices = [i for i, col in enumerate(feature_names) if col in ['tr_src', 'tr_dst']]
    numerical_features = [col for col in feature_names if col not in ['tr_src', 'tr_dst']]

    # --- Treinamento com Validação Cruzada ---
    print(f"[INFO] Iniciando treinamento com {N_SPLITS}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    lgbm_oof_preds = np.zeros(len(train_df))
    lgbm_test_preds = np.zeros(len(test_df))
    cat_oof_preds = np.zeros(len(train_df))
    cat_test_preds = np.zeros(len(test_df))

    lgbm_params = { 'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'n_estimators': 3000, 'seed': SEED, 'n_jobs': -1, 'verbose': -1, 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 8, 'num_leaves': 40, 'reg_alpha': 1.0, 'reg_lambda': 1.0, 'scale_pos_weight': 20, 'subsample': 0.7 }
    cat_params = { 'iterations': 3000, 'learning_rate': 0.03, 'depth': 8, 'loss_function': 'Logloss', 'eval_metric': 'AUC', 'scale_pos_weight': 20, 'random_seed': SEED, 'verbose': 0, 'thread_count': -1 }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all_df, y_all)):
        print(f"\n----- FOLD {fold+1}/{N_SPLITS} -----")
        
        # MODIFIED: Prepare separate data for LGBM (scaled) and CatBoost (unscaled)
        Xtr_df, Xval_df = X_all_df.iloc[train_idx], X_all_df.iloc[val_idx]
        ytr, yval = y_all[train_idx], y_all[val_idx]
        
        # --- Scaler for numerical features (for LGBM) ---
        scaler = StandardScaler().fit(Xtr_df[numerical_features].loc[ytr == 0])
        Xtr_scaled_num = scaler.transform(Xtr_df[numerical_features])
        Xval_scaled_num = scaler.transform(Xval_df[numerical_features])
        X_test_scaled_num = scaler.transform(X_test_df[numerical_features])

        # --- LightGBM (uses only scaled numerical features) ---
        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(Xtr_scaled_num, ytr, eval_set=[(Xval_scaled_num, yval)], callbacks=[lgb.early_stopping(150, verbose=False)])
        lgbm_oof_preds[val_idx] = lgbm.predict_proba(Xval_scaled_num)[:, 1]
        lgbm_test_preds += lgbm.predict_proba(X_test_scaled_num)[:, 1] / N_SPLITS
        
        # --- CatBoost (uses all features, with unscaled categorical ones) ---
        cat = CatBoostClassifier(**cat_params)
        cat.fit(Xtr_df, ytr, 
                eval_set=[(Xval_df, yval)],
                cat_features=categorical_features_indices,
                early_stopping_rounds=150, use_best_model=True)
        cat_oof_preds[val_idx] = cat.predict_proba(Xval_df)[:, 1]
        cat_test_preds += cat.predict_proba(X_test_df)[:, 1] / N_SPLITS

    print("\n[INFO] Treinamento CV finalizado.")

    # --- NEW: Stacking Meta-Model ---
    print("\n[INFO] Treinando meta-modelo de stacking...")
    
    # Create new training and test sets from the base models' predictions
    X_meta_train = np.vstack([lgbm_oof_preds, cat_oof_preds]).T
    X_meta_test = np.vstack([lgbm_test_preds, cat_test_preds]).T

    # Train a simple Logistic Regression model as the meta-model
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_train, y_all)

    # Get final predictions from the meta-model
    stacked_oof_preds = meta_model.predict_proba(X_meta_train)[:, 1]
    stacked_test_preds = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # --- Final Evaluation and Submission ---
    best_thr = calibrate_threshold_by_validation(stacked_oof_preds, y_all)
    ypred_oof = (stacked_oof_preds > best_thr).astype(int)
    ypred_test = (stacked_test_preds > best_thr).astype(int)
    
    print("\n[INFO] Relatório de Classificação Final do Stacking Ensemble (Out-of-Fold):")
    print(classification_report(y_all, ypred_oof, digits=4))
    
    sub = pd.DataFrame({"id": test_df["tr_id"].astype(int), "target": ypred_test.astype(int)})
    sub.to_csv("submission_stacking_final.csv", index=False)
    print(f"\n[INFO] Arquivo de submissão salvo: {os.path.abspath('submission_stacking_final.csv')}")
