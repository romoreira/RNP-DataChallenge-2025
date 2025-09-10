# autoencoder.py
# Autoencoder para reconstrução de features com skew/kurtosis
# Fluxo idêntico ao svdd_test8.py (com cache, split, calibração e submission)

import os, time, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import tensorflow as tf
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras import regularizers

SEED = 42
VAL_SIZE = 0.20
CACHE_DIR = "./_cache_feats"
CACHE_TAG = "AUTOENC"
USE_CACHE = True
CSV_ENGINE = "pyarrow"
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
NROWS = None

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =======================
# Leitura rápida de CSV
# =======================
def read_csv_fast(path, nrows=None, use_engine="pyarrow"):
    t0 = time.time()
    kw = dict(nrows=nrows)
    if use_engine == "pyarrow":
        try:
            df = pd.read_csv(path, engine="pyarrow", **kw)
            print(f"[INFO] read_csv (pyarrow) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
            return df
        except Exception as e:
            print(f"[WARN] pyarrow falhou ({e}), usando engine='c'")
    df = pd.read_csv(path, engine="c", **kw)
    print(f"[INFO] read_csv (c) OK: {path} | {df.shape} | {time.time()-t0:.1f}s")
    return df

# =======================
# Featurização (igual SVDD + skew/kurtosis)
# =======================
def fast_parse_array(s: str):
    if not isinstance(s, str) or len(s) < 2:
        return np.array([], dtype=np.float32)
    return np.fromstring(s[1:-1], sep=',', dtype=np.float32)

def featurize_fast(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import skew, kurtosis
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)

    means   = np.fromiter((a.mean() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds    = np.fromiter((a.std(ddof=0) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    skews   = np.fromiter((skew(a) if a.size > 1 else 0.0 for a in rtts), dtype=np.float32, count=n)
    kurts   = np.fromiter((kurtosis(a) if a.size > 1 else 0.0 for a in rtts), dtype=np.float32, count=n)

    out = pd.DataFrame({
        "rtt_mean": means,
        "rtt_std": stds,
        "rtt_skew": skews,
        "rtt_kurt": kurts,
        "tr_attempts": df["tr_attempts"].astype(np.float32),
        "total_probes_sent": df["total_probes_sent"].astype(np.float32),
        "total_replies_last_hop": df["total_replies_last_hop"].astype(np.float32),
        "seconds_since_start": df["seconds_since_start"].astype(np.float32),
    })
    eps = 1e-9
    out["success_rate"] = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

    print(f"[INFO] featurize_fast: {out.shape} | {time.time()-t0:.1f}s")
    return out

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if USE_CACHE and os.path.exists(cache_path):
        try:
            t0 = time.time()
            df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) == nrows):
                print(f"[INFO] Cache OK: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
                return df
            else:
                print("[INFO] Cache existe mas nrows difere, recalculando")
        except Exception as e:
            print(f"[WARN] Falha cache ({e}), recalculando")
    base = read_csv_fast(csv_path, nrows=nrows, use_engine=CSV_ENGINE)
    feats = featurize_fast(base)
    if USE_CACHE:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feats.to_parquet(cache_path, index=False)
        print(f"[INFO] Cache salvo em {cache_path}")
    return feats

# =======================
# Autoencoder
# =======================
def autoencoder_model(n_features):
    inputs = Input(shape=(1, n_features))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(1)(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(n_features))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

def calibrate_f1(loss, ytrue):
    best_thr, best_f1 = 0, -1
    for thr in np.linspace(loss.min(), loss.max(), 200):
        ypred = (loss > thr).astype(int)
        if len(np.unique(ypred)) < 2: continue
        f1 = f1_score(ytrue, ypred, average="macro")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

# =======================
# Fluxo principal
# =======================
if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    cache_test  = os.path.join(CACHE_DIR, f"test_feats_{CACHE_TAG}_{NROWS or 'ALL'}.parquet")

    # ----- Leitura CSV
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    test_df  = read_csv_fast(TEST_CSV,  nrows=NROWS, use_engine=CSV_ENGINE)
    y_all = train_df["route_changed"].astype(int).values

    # ----- Featurização
    X_all = maybe_cached_features(TRAIN_CSV, cache_train, nrows=NROWS)
    X_te  = maybe_cached_features(TEST_CSV, cache_test, nrows=NROWS)

    # ----- Split
    Xtr_df, Xval_df, ytr, yval = train_test_split(
        X_all, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )

    # ----- Normalização só nos normais
    scaler = StandardScaler().fit(Xtr_df[ytr==0])
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    Xte_s = scaler.transform(X_te)

    # reshape [samples, timesteps=1, features]
    Xtr = Xtr.reshape(Xtr.shape[0], 1, Xtr.shape[1])
    Xval = Xval.reshape(Xval.shape[0], 1, Xval.shape[1])
    Xte_s = Xte_s.reshape(Xte_s.shape[0], 1, Xte_s.shape[1])

    # ----- Modelo
    model = autoencoder_model(Xtr.shape[2])
    model.compile(optimizer="adam", loss="mae")
    model.summary()
    model.fit(
        Xtr[ytr==0], Xtr[ytr==0],
        epochs=50, batch_size=64,
        validation_split=0.1, verbose=2
    )

    # ----- Validação para threshold
    Xval_pred = model.predict(Xval)
    Xval_pred = Xval_pred.reshape(Xval_pred.shape[0], Xval_pred.shape[2])
    Xval_flat = Xval.reshape(Xval.shape[0], Xval.shape[2])
    loss_val = np.mean(np.abs(Xval_pred - Xval_flat), axis=1)

    thr, best_f1 = calibrate_f1(loss_val, yval)
    print(f"[INFO] Melhor threshold={thr:.6f} | F1-macro={best_f1:.4f}")

    ypred_val = (loss_val > thr).astype(int)
    print(confusion_matrix(yval, ypred_val))
    print(classification_report(yval, ypred_val, digits=4))

    # ----- Teste
    Xte_pred = model.predict(Xte_s)
    Xte_pred = Xte_pred.reshape(Xte_pred.shape[0], Xte_pred.shape[2])
    Xte_flat = Xte_s.reshape(Xte_s.shape[0], Xte_s.shape[2])
    loss_te = np.mean(np.abs(Xte_pred - Xte_flat), axis=1)
    ypred_te = (loss_te > thr).astype(int)

    # ----- Submission
    sub = pd.DataFrame({
        "id": test_df["tr_id"].astype(int),
        "target": ypred_te.astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print("[INFO] Arquivo salvo: submission.csv")

