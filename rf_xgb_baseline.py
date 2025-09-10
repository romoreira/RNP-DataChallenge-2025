import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ===============================
# Configurações e caminhos
# ===============================
SEED = 42
VAL_SIZE = 0.20
N_SAMPLE = 2_000_000   # Limite para evitar OOM (ajuste conforme sua RAM)
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
CACHE_DIR = "./_cache_feats"
CACHE_TRAIN = os.path.join(CACHE_DIR, f"train_feats_ALL.parquet")
CACHE_TEST = os.path.join(CACHE_DIR, f"test_feats_ALL.parquet")

def read_csv_fast(path, nrows=None):
    t0 = time.time()
    try:
        df = pd.read_csv(path, engine="pyarrow", nrows=nrows)
        print(f"[INFO] read_csv (pyarrow) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[WARN] pyarrow fail ({e}), fallback to engine='c'.")
        df = pd.read_csv(path, engine="c", nrows=nrows)
        print(f"[INFO] read_csv (c) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if os.path.exists(cache_path):
        t0 = time.time()
        df = pd.read_parquet(cache_path)
        if (nrows is None) or (len(df) == nrows):
            print(f"[INFO] Loaded from cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
            return df
        else:
            print("[INFO] Cache exists but nrows differ; recalculating.")
    else:
        print("[INFO] Cache not found, featurize CSV directly.")
        df = read_csv_fast(csv_path, nrows=nrows)
    return df

def calibrate_threshold(y_true, y_prob):
    prec, rec, thrs = precision_recall_curve(y_true, y_prob)
    f1s = 2*prec*rec/(prec+rec+1e-8)
    i = np.argmax(f1s)
    best_thr = thrs[i] if i < len(thrs) else 0.5
    best_f1 = f1s[i]
    print(f"[INFO] Best threshold: {best_thr:.6f} | F1-macro: {best_f1:.6f}")
    return best_thr, best_f1

if __name__ == "__main__":
    np.random.seed(SEED)

    # ===============================
    # Leitura dos CSVs e features
    # ===============================
    print("[INFO] Lendo CSVs e features ...")
    train_df = read_csv_fast(TRAIN_CSV)
    test_df = read_csv_fast(TEST_CSV)
    X_all = maybe_cached_features(TRAIN_CSV, CACHE_TRAIN)
    X_te = maybe_cached_features(TEST_CSV, CACHE_TEST)
    y_all = train_df["route_changed"].astype(int).values

    print("[INFO] Features treino:", X_all.shape, "| Teste:", X_te.shape)
    print("[INFO] Distribuição de classes treino:", np.bincount(y_all))

    # ===============================
    # Subamostragem para caber na RAM
    # ===============================
    if X_all.shape[0] > N_SAMPLE:
        print(f"[WARN] Muitos dados ({X_all.shape[0]}), usando apenas {N_SAMPLE} amostras aleatórias para treinamento.")
        idx = np.random.choice(X_all.shape[0], N_SAMPLE, replace=False)
        X_all = X_all.iloc[idx].reset_index(drop=True)
        y_all = y_all[idx]

    # ===============================
    # Split train/val
    # ===============================
    Xtr, Xval, ytr, yval = train_test_split(X_all, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)
    print("[INFO] Train:", Xtr.shape, "Val:", Xval.shape)

    # ===============================
    # Padronização dos dados
    # ===============================
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)
    X_all_s = scaler.transform(X_all)
    X_te_s = scaler.transform(X_te)

    # ===============================
    # RANDOM FOREST
    # ===============================
    print("\n========== RandomForest ==========")
    rf = RandomForestClassifier(
        n_estimators=50,          # Menos árvores para economizar RAM
        max_depth=12,             # Limita profundidade das árvores
        max_samples=500_000,      # Cada árvore só vê 500k amostras
        n_jobs=-1,
        class_weight="balanced",
        random_state=SEED
    )
    t0 = time.time()
    rf.fit(Xtr_s, ytr)
    print(f"[INFO] RF treinado em {time.time()-t0:.1f}s")

    yval_prob = rf.predict_proba(Xval_s)[:, 1]
    best_thr_rf, best_f1_rf = calibrate_threshold(yval, yval_prob)
    yval_pred = (yval_prob > best_thr_rf).astype(int)
    print(classification_report(yval, yval_pred, digits=4))

    # ===============================
    # XGBOOST
    # ===============================
    print("\n========== XGBoost ==========")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(),
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    t0 = time.time()
    xgb_model.fit(Xtr_s, ytr)
    print(f"[INFO] XGB treinado em {time.time()-t0:.1f}s")

    yval_prob_xgb = xgb_model.predict_proba(Xval_s)[:, 1]
    best_thr_xgb, best_f1_xgb = calibrate_threshold(yval, yval_prob_xgb)
    yval_pred_xgb = (yval_prob_xgb > best_thr_xgb).astype(int)
    print(classification_report(yval, yval_pred_xgb, digits=4))

    # ===============================
    # Escolha do melhor modelo
    # ===============================
    print("\nEscolha automática do melhor modelo para submissão com base no F1-macro valid.")
    if best_f1_xgb >= best_f1_rf:
        final_model = xgb_model
        final_thr = best_thr_xgb
        model_name = 'xgboost'
    else:
        final_model = rf
        final_thr = best_thr_rf
        model_name = 'random_forest'
    print(f"[INFO] Usando modelo: {model_name} | Threshold: {final_thr:.6f}")

    # ===============================
    # Inferência no teste e submissão
    # ===============================
    print("[INFO] Inferência no teste e salvando submission.csv ...")
    ytest_prob = final_model.predict_proba(X_te_s)[:, 1]
    ytest_pred = (ytest_prob > final_thr).astype(int)
    print("Distribuição predita teste:", np.bincount(ytest_pred))

    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ytest_pred.astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print(f"[INFO] Arquivo salvo: {os.path.abspath('submission.csv')}")
