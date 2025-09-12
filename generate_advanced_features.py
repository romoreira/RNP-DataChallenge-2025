# generate_advanced_features.py
# Script dedicado para criar o novo conjunto de features com
# a feature de contexto (is_dst_1) e as features de janela deslizante (rolling window).

import os
import time
import numpy as np
import pandas as pd

# =======================
# Configurações principais
# =======================
SEED = 42

# --- Configurações de Entrada ---
TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"
CSV_ENGINE = "pyarrow"
NROWS = None

# --- Configurações de Saída ---
# Novo diretório para não sobrescrever as features antigas
NEW_CACHE_DIR = "./_cache_feats_ROLLING"
NEW_CACHE_TAG = "ROLLING1"

# ================================================================
# Funções de utilidade e a nova função de featurização
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

def featurize_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de featurização atualizada com as novas features de contexto e janela deslizante.
    """
    t0 = time.time()
    rtts = df["all_rtts"].apply(fast_parse_array)
    n = len(rtts)
    
    # --- Features básicas (como antes) ---
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

    # --- NOVA FEATURE DE CONTEXTO ---
    # Ação 1: Isolar o "Efeito Destino 1"
    if 'tr_dst' in df.columns:
        out['is_dst_1'] = (df['tr_dst'] == 1).astype(np.float32)
    else:
        out['is_dst_1'] = 0.0

    # --- Features Temporais (Deltas e Janelas Deslizantes) ---
    required = {"tr_src", "tr_dst", "seconds_since_start"}
    if required.issubset(df.columns):
        # Usamos um DataFrame temporário para ordenação e cálculos de grupo
        tmp = out[["rtt_mean", "rtt_p90", "success_rate", "replies_per_attempt", "seconds_since_start"]].copy()
        tmp["tr_src"] = df["tr_src"]
        tmp["tr_dst"] = df["tr_dst"]
        tmp["__rowid__"] = np.arange(len(tmp), dtype=np.int64)
        tmp.sort_values(["tr_src", "tr_dst", "seconds_since_start"], kind="mergesort", inplace=True)

        # Features Delta (como antes)
        def add_delta(col, clip_ratio=(0.0, 10.0)):
            prev = tmp.groupby(["tr_src", "tr_dst"])[col].shift(1)
            tmp[f"delta_{col}"] = (tmp[col] - prev).astype(np.float32)
            ratio = tmp[col] / (prev + eps)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tmp[f"ratio_{col}"] = ratio.clip(*clip_ratio).astype(np.float32)
        for base_col in ["rtt_mean", "rtt_p90", "success_rate", "replies_per_attempt"]:
            add_delta(base_col)
        
        # --- NOVAS FEATURES DE JANELA DESLIZANTE ---
        # Ação 2: Criar features robustas a picos de latência
        window_size = 5
        rolling_mean = tmp.groupby(['tr_src', 'tr_dst'])['rtt_mean'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=2).mean() # Usamos shift(1) para não vazar info do ponto atual
        )
        rolling_std = tmp.groupby(['tr_src', 'tr_dst'])['rtt_mean'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=2).std()
        )
        tmp['zscore_vs_rolling'] = (tmp['rtt_mean'] - rolling_mean) / (rolling_std + eps)
        tmp['ratio_vs_rolling'] = tmp['rtt_mean'] / (rolling_mean + eps)

        # Outras features contextuais
        tmp["time_since_prev"] = tmp.groupby(["tr_src", "tr_dst"])["seconds_since_start"].diff().fillna(0.0).astype(np.float32)
        tmp["is_first_obs"] = (tmp.groupby(["tr_src", "tr_dst"]).cumcount() == 0).astype(np.float32)

        # Juntar tudo
        tmp.sort_values("__rowid__", inplace=True)
        tmp.drop(columns=['__rowid__', 'tr_src', 'tr_dst'], inplace=True)
        
        # Preencher NaNs que podem ter sido gerados
        tmp.fillna(0.0, inplace=True)
        tmp.replace([np.inf, -np.inf], 0.0, inplace=True)
        
        # Lista final de colunas a serem adicionadas
        new_cols = [col for col in tmp.columns if col not in out.columns]
        out[new_cols] = tmp[new_cols].values
    else:
        # Fallback para o dataset de teste que não tem todas as colunas
        print("[WARN] Colunas de rota/tempo não encontradas. Features temporais não serão criadas.")

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    print(f"[INFO] featurize_advanced: {out.shape} | {time.time()-t0:.1f}s")
    return out

# =======================
# Fluxo Principal de Geração de Features
# =======================
if __name__ == "__main__":
    set_seed(SEED)
    
    # Criar o novo diretório de cache
    os.makedirs(NEW_CACHE_DIR, exist_ok=True)
    
    # --- Processar dados de Treino ---
    print("\n[INFO] --- Processando dados de TREINO ---")
    train_df = read_csv_fast(TRAIN_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    X_train_advanced = featurize_advanced(train_df)
    
    # Salvar o novo parquet de treino
    train_output_path = os.path.join(NEW_CACHE_DIR, f"train_feats_{NEW_CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    X_train_advanced.to_parquet(train_output_path, index=False)
    print(f"[INFO] Features de treino avançadas salvas em: {train_output_path}")

    # --- Processar dados de Teste ---
    print("\n[INFO] --- Processando dados de TESTE ---")
    test_df = read_csv_fast(TEST_CSV, nrows=NROWS, use_engine=CSV_ENGINE)
    X_test_advanced = featurize_advanced(test_df)

    # Salvar o novo parquet de teste
    test_output_path = os.path.join(NEW_CACHE_DIR, f"test_feats_{NEW_CACHE_TAG}_{NROWS or 'ALL'}.parquet")
    X_test_advanced.to_parquet(test_output_path, index=False)
    print(f"[INFO] Features de teste avançadas salvas em: {test_output_path}")
    
    print("\n[INFO] Processo de geração de features concluído com sucesso!")
