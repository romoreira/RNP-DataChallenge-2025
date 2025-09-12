# analyze_anomaly_scores.py
# Script completo para treinar um modelo de detecção de anomalia (Isolation Forest)
# e analisar a distribuição de seus scores entre os grupos de erro do modelo principal.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configurações ---
SEED = 42
VAL_SIZE = 0.20
CACHE_DIR = "./_cache_feats_ROLLING"
CACHE_TAG = "ROLLING1"
TRAIN_CSV = "dataset/train.csv"
# Reutilize os melhores parâmetros encontrados para o ROLLING1 (mesmo que parciais)
# Se não tiver, use os do DELTA1 como ponto de partida para identificar os erros.
best_params = {
    'colsample_bytree': 0.8049, 'learning_rate': 0.0876, 'max_depth': 9,
    'num_leaves': 59, 'reg_alpha': 2.639, 'reg_lambda': 0.435,
    'scale_pos_weight': 5, 'subsample': 0.859
}
# --- Fim Configurações ---

def set_seed(seed=42):
    np.random.seed(seed)

def read_csv_fast(path, usecols=None):
    try:
        df = pd.read_csv(path, engine="pyarrow", usecols=usecols)
        return df
    except Exception:
        df = pd.read_csv(path, engine="c", usecols=usecols)
        return df

def load_features_from_cache(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Arquivo de cache não encontrado: {cache_path}. Execute o script 'generate_advanced_features.py' primeiro.")
    return pd.read_parquet(cache_path)

if __name__ == "__main__":
    set_seed(SEED)

    # FASE 1: Replicar o ambiente para encontrar os FPs e TPs
    print("[INFO] FASE 1: Replicando ambiente para identificar Falsos Positivos (FP) e Verdadeiros Positivos (TP)...")
    train_df = read_csv_fast(TRAIN_CSV, usecols=['route_changed'])
    cache_path = f"{CACHE_DIR}/train_feats_{CACHE_TAG}_ALL.parquet"
    X_all_df = load_features_from_cache(cache_path)
    y_all = train_df["route_changed"].astype(int).values

    Xtr_df, Xval_df, ytr, yval = train_test_split(X_all_df, y_all, test_size=VAL_SIZE, random_state=SEED, stratify=y_all)

    print(f"[INFO] Treino: {Xtr_df.shape}, Validação: {Xval_df.shape}")

    scaler = StandardScaler().fit(Xtr_df[ytr == 0])
    Xtr, Xval = scaler.transform(Xtr_df), scaler.transform(Xval_df)
    
    # Adicionando parâmetros fixos para o LGBM de identificação
    lgbm_params = best_params.copy()
    lgbm_params.update({'objective': 'binary', 'seed': SEED, 'n_jobs': -1})

    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(Xtr, ytr)
    # Usando predict com limiar padrão 0.5 para simplicidade na identificação dos grupos
    ypred_val = model.predict(Xval)

    fp_mask = (ypred_val == 1) & (yval == 0)
    tp_mask = (ypred_val == 1) & (yval == 1)
    tn_mask = (ypred_val == 0) & (yval == 0)
    print(f"[INFO] Grupos identificados: {fp_mask.sum()} FPs, {tp_mask.sum()} TPs, {tn_mask.sum()} TNs.")


    # FASE 2: Treinar Isolation Forest e obter scores
    print("\n[INFO] FASE 2: Treinando Isolation Forest nos dados normais de treino...")
    # Usar um subconjunto dos dados normais se for muito grande para acelerar o processo
    Xtr_normal_samples = Xtr[ytr == 0]
    sample_size = min(len(Xtr_normal_samples), 500000) # Limita a 500k amostras para o fit
    np.random.shuffle(Xtr_normal_samples)
    Xtr_normal_subset = Xtr_normal_samples[:sample_size]
    
    iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=SEED, n_jobs=-1)
    iso_forest.fit(Xtr_normal_subset)

    print("[INFO] Calculando scores de anomalia para todo o conjunto de validação...")
    # O score é invertido (-1 * decision_function) para que valores mais altos signifiquem MAIS anomalia
    anomaly_scores = -1 * iso_forest.decision_function(Xval)


    # FASE 3: Plotar a distribuição dos scores
    print("\n[INFO] FASE 3: Gerando gráfico de distribuição dos scores...")
    results_df = pd.DataFrame({
        'score': anomaly_scores,
        'type': np.select([fp_mask, tp_mask, tn_mask], ['Falso Positivo (FP)', 'Verdadeiro Positivo (TP)', 'Verdadeiro Negativo (TN)'], 'Outro')
    })
    results_df = results_df[results_df['type'] != 'Outro']

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='type', y='score', data=results_df, order=['Verdadeiro Negativo (TN)', 'Falso Positivo (FP)', 'Verdadeiro Positivo (TP)'])
    plt.title('Distribuição dos Scores de Anomalia (Isolation Forest) por Grupo', fontsize=16)
    plt.ylabel('Score de Anomalia (Quanto maior, mais anômalo)', fontsize=12)
    plt.xlabel('')
    plt.xticks(fontsize=12)
    plt.tight_layout()
    output_path = "anomaly_score_analysis.png"
    plt.savefig(output_path)
    print(f"[INFO] Gráfico salvo em: {os.path.abspath(output_path)}")
