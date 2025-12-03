import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import sys

# Configuração
INPUT_FILE = "results/resultado_experimento_metricas.csv"
MY_MODEL_NAME = "Ensemble_Stacking" # Nome exato que aparece no gráfico

print(f"[INFO] Lendo {INPUT_FILE}...")

# 1. Leitura Robusta (Ignora linhas corrompidas)
try:
    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
except TypeError:
    df = pd.read_csv(INPUT_FILE, error_bad_lines=False)

print(f"[INFO] Linhas carregadas: {len(df)}")

# 2. Filtragem e Limpeza
# Removemos modelos irrelevantes se houver sujeira
valid_models = ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'KNN', 'DecisionTree', 'Dummy', MY_MODEL_NAME]
df = df[df['Modelo'].isin(valid_models)]

# Pivota para formato pareado (Linha=Round, Coluna=Modelo)
pivot_df = df.pivot(index='Iteracao', columns='Modelo', values='F1_Score')

# Remove rodadas que não tenham todos os modelos (para garantir N=10 pareado perfeito)
pivot_df = pivot_df.dropna()

print(f"[INFO] Rodadas válidas completas (N): {len(pivot_df)}")
print("-" * 30)
print("MÉDIAS REAIS (Confira com o Gráfico):")
print(pivot_df.mean().sort_values(ascending=False))
print("-" * 30)

# ==============================================================================
# 3. NOVOS TESTES ESTATÍSTICOS
# ==============================================================================

# Friedman
stat, p_friedman = friedmanchisquare(*[pivot_df[col].values for col in pivot_df.columns])

# Wilcoxon (One-Tailed: Ensemble > Outros)
results = []
others = [m for m in pivot_df.columns if m != MY_MODEL_NAME]

for opponent in others:
    # alternative='greater' -> Testa se Ensemble > Oponente
    s, p = wilcoxon(pivot_df[MY_MODEL_NAME], pivot_df[opponent], alternative='greater')
    results.append({
        'Opponent': opponent,
        'P-value': p,
        'Significant': p < 0.05
    })

# ==============================================================================
# 4. GERAR TABELA LATEX FINAL
# ==============================================================================
print("\n=== COPIE ESTA TABELA PARA O SEU PAPER ===")
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Statistical evaluation (Wilcoxon Signed-Rank Test, one-tailed) comparing the proposed Stacking Ensemble against baseline models.}")
print(r"\label{tab:wilcoxon_results}")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"\textbf{Comparison} & \textbf{P-value} & \textbf{Sig. ($\alpha=0.05$)} \\")
print(r"\midrule")

for res in results:
    # Formatação bonita
    if res['P-value'] < 0.001:
        p_str = "$< 0.001$"
    else:
        p_str = f"{res['P-value']:.4f}"
    
    sig_str = r"\textbf{Yes}" if res['Significant'] else "No"
    
    print(f"Stacking vs. {res['Opponent']} & {p_str} & {sig_str} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")