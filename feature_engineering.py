# inspecao_erros_completo.py
# Script completo e autônomo para análise de erros do modelo final no conjunto de validação.

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================================================================
# Funções de utilidade, featurização e pós-processamento
# Todas as funções do seu script original estão incluídas aqui para garantir consistência.
# ================================================================
SEED = 42

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
    means = np.fromiter((a.mean() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    stds = np.fromiter((a.std(ddof=0) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    medians = np.fromiter((np.median(a) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    p90s = np.fromiter((np.quantile(a,0.9) if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    mins = np.fromiter((a.min() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    maxs = np.fromiter((a.max() if a.size else 0.0 for a in rtts), dtype=np.float32, count=n)
    lens = np.fromiter((a.size for a in rtts), dtype=np.float32, count=n)
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
    out["success_rate"] = out["total_replies_last_hop"] / (out["total_probes_sent"] + eps)
    out["loss_rate"] = 1.0 - out["success_rate"]
    out["replies_per_attempt"] = out["total_replies_last_hop"] / (out["tr_attempts"] + eps)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    required = {"tr_src", "tr_dst", "seconds_since_start"}
    if required.issubset(df.columns):
        tmp = out[["rtt_mean", "rtt_p90", "success_rate", "replies_per_attempt", "seconds_since_start"]].copy()
        tmp["tr_src"] = df["tr_src"]
        tmp["tr_dst"] = df["tr_dst"]
        tmp["__rowid__"] = np.arange(len(tmp), dtype=np.int64)
        tmp.sort_values(["tr_src", "tr_dst", "seconds_since_start"], kind="mergesort", inplace=True)
        def add_delta(col, clip_ratio=(0.0, 10.0)):
            prev = tmp.groupby(["tr_src", "tr_dst"])[col].shift(1)
            dcol = f"delta_{col}"
            rcol = f"ratio_{col}"
            tmp[dcol] = (tmp[col] - prev).astype(np.float32)
            ratio = tmp[col] / prev.replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tmp[rcol] = ratio.clip(*clip_ratio).astype(np.float32)
        for base_col in ["rtt_mean", "rtt_p90", "success_rate", "replies_per_attempt"]:
            add_delta(base_col)
        tmp["time_since_prev"] = tmp.groupby(["tr_src", "tr_dst"])["seconds_since_start"].diff().fillna(0.0).astype(np.float32)
        tmp["is_first_obs"] = (tmp.groupby(["tr_src", "tr_dst"]).cumcount() == 0).astype(np.float32)
        tmp.sort_values("__rowid__", inplace=True)
        new_cols = [
            "delta_rtt_mean", "ratio_rtt_mean", "delta_rtt_p90", "ratio_rtt_p90",
            "delta_success_rate", "ratio_success_rate", "delta_replies_per_attempt", "ratio_replies_per_attempt",
            "time_since_prev", "is_first_obs"
        ]
        out[new_cols] = tmp[new_cols].values
    else:
        out["time_since_prev"] = 0.0
        out["is_first_obs"] = 1.0
        for c in ["rtt_mean", "rtt_p90", "success_rate", "replies_per_attempt"]:
            out[f"delta_{c}"] = 0.0
            out[f"ratio_{c}"] = 1.0
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    print(f"[INFO] featurize_fast: {out.shape} | {time.time()-t0:.1f}s")
    return out

def maybe_cached_features(csv_path, cache_path, nrows=None):
    if os.path.exists(cache_path):
        try:
            t0 = time.time()
            df = pd.read_parquet(cache_path)
            if (nrows is None) or (len(df) == (nrows or len(df))):
                print(f"[INFO] Carregado do cache: {cache_path} | {df.shape} | {time.time()-t0:.1f}s")
                return df
        except Exception as e:
            print(f"[WARN] Falha ao ler cache ({e}). Recriando.")
    base = read_csv_fast(csv_path, nrows=nrows)
    feats = featurize_fast(base)
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
    g_mean = float(s_norm.mean()) if s_norm.size > 0 else 0.5
    g_std = float(s_norm.std() + eps_std) if s_norm.size > 0 else 1.0
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
    total_pos = int(cum_pos[-1]) if len(cum_pos) > 0 else 0
    TP = total_pos - cum_pos
    FP = (len(y) - total_pos) - cum_neg
    FN = cum_pos
    prec1 = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) > 0)
    rec1 = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) > 0)
    f1_1 = np.divide(2 * prec1 * rec1, (prec1 + rec1), out=np.zeros_like(prec1), where=(prec1 + rec1) > 0)
    if len(f1_1) == 0: return 0.5
    j = int(np.argmax(f1_1))
    best_thr = float(s[j]) if len(s) > 0 else 0.5
    return best_thr

# =======================
# Fluxo de Inspeção
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # --- FASE 1: Replicar o cenário do modelo final ---
    print("[INFO] FASE 1: Replicando o ambiente do modelo final...")
    
    # Configurações do modelo
    VAL_SIZE = 0.20
    CACHE_DIR = "./_cache_feats"
    CACHE_TAG = "DELTA1"
    TRAIN_CSV = "dataset/train.csv"
    best_params = {
        'colsample_bytree': 0.8049, 'learning_rate': 0.0876, 'max_depth': 9,
        'num_leaves': 59, 'reg_alpha': 2.639, 'reg_lambda': 0.435,
        'scale_pos_weight': 5, 'subsample': 0.859
    }
    
    # Carregar dados e features
    cache_train = os.path.join(CACHE_DIR, f"train_feats_{CACHE_TAG}_ALL.parquet")
    train_df = read_csv_fast(TRAIN_CSV)
    X_all_df = maybe_cached_features(TRAIN_CSV, cache_train)
    y_all = train_df["route_changed"].astype(int).values
    rk_all = make_route_keys(train_df)
    
    # Fazer o mesmo split
    Xtr_df, Xval_df, ytr, yval, rk_tr, rk_val = train_test_split(
        X_all_df, y_all, rk_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all
    )
    
    # Aplicar o mesmo Scaler
    scaler = StandardScaler().fit(Xtr_df[ytr == 0])
    Xtr = scaler.transform(Xtr_df)
    Xval = scaler.transform(Xval_df)
    
    # Treinar o mesmo modelo
    print("[INFO] Treinando o modelo para obter as predições de validação...")
    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt', n_estimators=2500,
        seed=SEED, n_jobs=-1, verbose=-1, **best_params
    )
    model.fit(Xtr, ytr,
              eval_set=[(Xval, yval)], eval_metric='auc',
              callbacks=[lgb.early_stopping(150, verbose=False)])

    # Gerar as mesmas predições
    scores_val_raw = model.predict_proba(Xval)[:, 1]
    mu_map, sd_map, g_mean, g_std = compute_group_norm_stats(scores_val_raw, yval, rk_val)
    scores_val_g = apply_group_zscore(scores_val_raw, rk_val, mu_map, sd_map, g_mean, g_std)
    best_thr = calibrate_threshold_by_validation(scores_val_g, yval)
    ypred_val = (scores_val_g > best_thr).astype(int)

    print("[INFO] Ambiente replicado com sucesso. Iniciando análise de erros...")
    print(classification_report(yval, ypred_val, digits=4))

    # --- FASE 2: Isolar os grupos de interesse ---
    
    # Máscaras para cada grupo
    fp_mask = (ypred_val == 1) & (yval == 0) # Falsos Positivos
    tp_mask = (ypred_val == 1) & (yval == 1) # Verdadeiros Positivos
    tn_mask = (ypred_val == 0) & (yval == 0) # Verdadeiros Negativos
    
    # DataFrames de features para cada grupo
    X_fp_df = Xval_df[fp_mask]
    X_tp_df = Xval_df[tp_mask]
    X_tn_df = Xval_df[tn_mask]
    
    # DataFrames originais (para info de rota)
    original_fp_df = train_df.loc[X_fp_df.index]
    original_tp_df = train_df.loc[X_tp_df.index]
    
    # Scores para cada grupo
    scores_raw_fp = scores_val_raw[fp_mask]
    scores_raw_tp = scores_val_raw[tp_mask]

    # --- FASE 3: Análise e Relatórios ---

    print("\n\n" + "="*80)
    print(" " * 28 + "INÍCIO DA ANÁLISE DE ERROS")
    print("="*80 + "\n")

    print(f"[INFO] Total de Falsos Positivos (FP): {len(X_fp_df)}")
    print(f"[INFO] Total de Verdadeiros Positivos (TP): {len(X_tp_df)}")
    print(f"[INFO] Total de Verdadeiros Negativos (TN): {len(X_tn_df)}\n")

    print("\n--- ANÁLISE 1: COMPARAÇÃO ESTATÍSTICA DAS FEATURES ---\n")
    print("--> Objetivo: Ver se os FPs têm características diferentes dos TPs ou TNs.\n")
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("--- Estatísticas dos FALSOS POSITIVOS (FP) ---")
        print(X_fp_df.describe().T)
        print("\n" + "-"*50 + "\n")
        
        print("--- Estatísticas dos VERDADEIROS POSITIVOS (TP) ---")
        print(X_tp_df.describe().T)
        print("\n" + "-"*50 + "\n")

        print("--- Estatísticas dos VERDADEIROS NEGATIVOS (TN) - População Normal ---")
        print(X_tn_df.describe().T)
        print("\n" + "-"*50 + "\n")

    print("\n--- ANÁLISE 2: ROTAS PROBLEMÁTICAS ---\n")
    print("--> Objetivo: Identificar se poucas rotas são responsáveis pela maioria dos erros.\n")
    print("--- Top 15 rotas com mais FALSOS POSITIVOS ---")
    print(original_fp_df.groupby(['tr_src', 'tr_dst']).size().nlargest(15))
    print("\n" + "-"*50 + "\n")

    print("\n--- ANÁLISE 3: IMPACTO DE FEATURES CONTEXTUAIS ---\n")
    print("--> Objetivo: Verificar se os erros acontecem em situações específicas.\n")
    if len(X_fp_df) > 0:
        fp_first_obs_count = X_fp_df['is_first_obs'].sum()
        print(f"Percentual de FPs que são a primeira medição da rota: {fp_first_obs_count / len(X_fp_df) * 100:.2f}% ({int(fp_first_obs_count)} de {len(X_fp_df)})")
    if len(X_tp_df) > 0:
        tp_first_obs_count = X_tp_df['is_first_obs'].sum()
        print(f"Percentual de TPs que são a primeira medição da rota: {tp_first_obs_count / len(X_tp_df) * 100:.2f}% ({int(tp_first_obs_count)} de {len(X_tp_df)})")
    print("\n" + "-"*50 + "\n")

    print("\n--- ANÁLISE 4: DISTRIBUIÇÃO DOS SCORES BRUTOS DO MODELO ---\n")
    print("--> Objetivo: Ver se o modelo está 'confiante' nos seus erros.\n")
    print("--- Estatísticas dos scores (probabilidades) para FALSOS POSITIVOS ---")
    print(pd.Series(scores_raw_fp).describe())
    print("\n--- Estatísticas dos scores (probabilidades) para VERDADEIROS POSITIVOS ---")
    print(pd.Series(scores_raw_tp).describe())
    print("\n" + "="*80)
    
    print("\n\n--- O QUE PROCURAR NOS RESULTADOS ---\n")
    print("1. Na ANÁLISE 1: Compare as colunas 'mean' e 'std' entre os 3 grupos. Os FPs têm 'rtt_std' ou 'delta_rtt_mean' maiores que os TNs? As features dos FPs se parecem mais com as dos TPs ou dos TNs?")
    print("2. Na ANÁLISE 2: Alguma rota se destaca absurdamente? Se uma única rota é responsável por muitos erros, ela pode ter um comportamento anômalo que não é uma mudança real.")
    print("3. Na ANÁLISE 3: Se o percentual de FPs com 'is_first_obs'=1 for alto, significa que o modelo erra muito quando não tem histórico. Uma possível ação é tratar essas primeiras medições de forma diferente.")
    print("4. Na ANÁLISE 4: Compare a distribuição dos scores. Se a média/mediana dos scores dos FPs for alta (ex: > 0.8), o modelo está muito confiante nos erros. Se for mais baixa, os erros são 'limítrofes' e mais fáceis de corrigir.")
