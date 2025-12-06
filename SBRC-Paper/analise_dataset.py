import pandas as pd
import os

# ================= CONFIGURAÇÃO =================
TRAIN_PATH = '../dataset/train.csv'
TEST_PATH = '../dataset/test.csv'
TARGET_COLUMN = 'route_changed'  # <--- CORRIGIDO
# ================================================

def print_header(title):
    print(f"\n{'='*40}")
    print(f" {title}")
    print(f"{'='*40}")

def analyze_train(df):
    print_header("ANÁLISE DETALHADA: CONJUNTO DE TREINO")
    
    # 1. Distribuição de Classes
    counts = df[TARGET_COLUMN].value_counts()
    percents = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({'Total': counts, 'Porcentagem (%)': percents.round(4)})
    print("\n--- Distribuição de Classes (Target: route_changed) ---")
    print(dist_df)
    
    # 2. Cálculo do Ratio de Desbalanceamento
    # Assumindo que 0 é a maioria e 1 é a anomalia/mudança
    try:
        majority = counts.max()
        minority = counts.min()
        ratio = majority / minority
        print(f"\n• Imbalance Ratio: 1:{ratio:.2f}")
        print(f"  (Existe 1 caso de 'route_changed' para cada {ratio:.2f} casos normais)")
    except:
        print("Erro ao calcular ratio (possivelmente apenas uma classe presente).")

    # 3. Estatísticas temporais ou de RTT (para enriquecer o texto)
    # Vamos pegar a média de probes para ter uma noção de esforço da rede
    if 'total_probes_sent' in df.columns:
        avg_probes = df['total_probes_sent'].mean()
        print(f"\n• Média de Probes enviados por traceroute: {avg_probes:.2f}")

def main():
    if not os.path.exists(TRAIN_PATH):
        print("❌ Erro: train.csv não encontrado.")
        return

    # Ler apenas colunas essenciais para ser mais rápido, se o arquivo for gigante
    # Se der erro de memória, avise.
    print("Lendo arquivos... aguarde.")
    df_train = pd.read_csv(TRAIN_PATH)
    
    print(f"Dataset carregado. Linhas: {len(df_train)}")
    
    if TARGET_COLUMN in df_train.columns:
        analyze_train(df_train)
    else:
        print(f"❌ Erro: Coluna '{TARGET_COLUMN}' ainda não encontrada no Treino.")

    # Análise rápida do Teste apenas para confirmar dimensões
    if os.path.exists(TEST_PATH):
        df_test = pd.read_csv(TEST_PATH)
        print_header("RESUMO: CONJUNTO DE TESTE")
        print(f"• Total de Instâncias para Predição: {len(df_test)}")
        print(f"• Proporção Treino/Teste: {len(df_train)/(len(df_train)+len(df_test))*100:.1f}% / {len(df_test)/(len(df_train)+len(df_test))*100:.1f}%")

if __name__ == "__main__":
    main()