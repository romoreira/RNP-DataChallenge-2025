# generate_final_features.py
# Script para adicionar a "golden feature" (score de anomalia) ao dataset.

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Configurações ---
SEED = 42
# Diretório de entrada com as features ROLLING1
INPUT_CACHE_DIR = "./_cache_feats_ROLLING"
INPUT_CACHE_TAG = "ROLLING1"
# Diretório de saída para as features finais
OUTPUT_CACHE_DIR = "./_cache_feats_FINAL"
OUTPUT_CACHE_TAG = "FINAL1"
# Arquivo CSV original para obter o target
TRAIN_CSV = "dataset/train.csv"
# --- Fim Configurações ---

def load_data(cache_dir, cache_tag, csv_path):
    print("[INFO] Lendo dados...")
    train_feats_path = os.path.join(cache_dir, f"train_feats_{cache_tag}_ALL.parquet")
    test_feats_path = os.path.join(cache_dir, f"test_feats_{cache_tag}_ALL.parquet")
    
    X_all = pd.read_parquet(train_feats_path)
    X_test = pd.read_parquet(test_feats_path)
    
    train_df = pd.read_csv(csv_path, usecols=['route_changed'])
    y_all = train_df['route_changed'].astype(int).values
    
    print(f"[INFO] Dados de treino carregados: {X_all.shape}")
    print(f"[INFO] Dados de teste carregados: {X_test.shape}")
    
    return X_all, X_test, y_all

if __name__ == "__main__":
    np.random.seed(SEED)
    
    # 1. Carregar os dados com as features ROLLING1
    X_all, X_test, y_all = load_data(INPUT_CACHE_DIR, INPUT_CACHE_TAG, TRAIN_CSV)
    
    # 2. Preparar os dados para o Isolation Forest
    # É crucial treinar o scaler e o iForest em TODOS os dados de treino normais
    print("\n[INFO] Preparando Scaler e Isolation Forest...")
    normal_data_mask = (y_all == 0)
    X_all_normal = X_all[normal_data_mask]
    
    # Scaler treinado apenas nos dados normais
    scaler = StandardScaler().fit(X_all_normal)
    X_all_normal_scaled = scaler.transform(X_all_normal)
    
    # Treinar o Isolation Forest
    print("[INFO] Treinando Isolation Forest em todos os dados normais de treino...")
    iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=SEED, n_jobs=-1)
    # Usar um subconjunto se a memória for um problema, mas idealmente usar tudo
    sample_size = min(len(X_all_normal_scaled), 1000000)
    idx = np.random.choice(len(X_all_normal_scaled), sample_size, replace=False)
    iso_forest.fit(X_all_normal_scaled[idx])
    
    # 3. Gerar a Golden Feature para os datasets completos
    print("[INFO] Gerando a 'golden feature' (anomaly_score) para treino e teste...")
    
    # Escalar todos os dados com o scaler já treinado
    X_all_scaled = scaler.transform(X_all)
    X_test_scaled = scaler.transform(X_test)
    
    # Adicionar a nova feature
    # Lembre-se: scores mais altos = mais anômalo
    X_all['anomaly_score'] = -1 * iso_forest.decision_function(X_all_scaled)
    X_test['anomaly_score'] = -1 * iso_forest.decision_function(X_test_scaled)
    
    print("[INFO] Estatísticas do novo 'anomaly_score':")
    print(X_all['anomaly_score'].describe())
    
    # 4. Salvar os novos datasets finais
    os.makedirs(OUTPUT_CACHE_DIR, exist_ok=True)
    train_output_path = os.path.join(OUTPUT_CACHE_DIR, f"train_feats_{OUTPUT_CACHE_TAG}_ALL.parquet")
    test_output_path = os.path.join(OUTPUT_CACHE_DIR, f"test_feats_{OUTPUT_CACHE_TAG}_ALL.parquet")
    
    X_all.to_parquet(train_output_path, index=False)
    X_test.to_parquet(test_output_path, index=False)
    
    print(f"\n[SUCESSO] Features finais salvas em '{OUTPUT_CACHE_DIR}' com a tag '{OUTPUT_CACHE_TAG}'.")
    print(f"Novo shape dos dados de treino: {X_all.shape}")
