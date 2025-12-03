import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO DE ESTILO ACM (High Quality)
# ==============================================================================
# Define fontes serifadas (estilo LaTeX/ACM) e tamanho 16
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# ==============================================================================
# 1. CARREGAR DADOS
# ==============================================================================
# Substitua pelo caminho real se necessário
csv_path = "results/resultado_experimento_metrics.csv" 

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # Dados de exemplo baseados no seu snippet para teste
    data = {
        'Iteracao': [1]*8 + [2]*8,
        'Modelo': ['Dummy', 'KNN', 'DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'Ensemble_Stacking']*2,
        'F1_Score': [0.0, 0.63, 0.75, 0.82, 0.84, 0.83, 0.85, 0.88, 
                     0.0, 0.64, 0.74, 0.83, 0.85, 0.84, 0.86, 0.89], # Exemplo
        'Total_Latency_ms_per_1k': [134, 140, 135, 160, 150, 145, 180, 250]*2,
        'Train_Time_sec': [0.5, 2.0, 5.0, 15.0, 20.0, 10.0, 40.0, 80.0]*2
    }
    df = pd.DataFrame(data)
    print("[AVISO] Usando dados de exemplo. Certifique-se de ter o CSV real.")

# Ordenar modelos por F1 Score mediano para o gráfico ficar bonito
order = df.groupby('Modelo')['F1_Score'].median().sort_values().index.tolist()

# Cores profissionais (Colorblind friendly palette)
palette = sns.color_palette("viridis", len(order))
model_colors = dict(zip(order, palette))

# ==============================================================================
# GRÁFICO 1: DISTRIBUIÇÃO DO F1-SCORE (BOXPLOT)
# Mostra estabilidade e performance
# ==============================================================================
plt.figure(figsize=(10, 6))
sns.boxplot(x='F1_Score', y='Modelo', data=df, order=order, palette=palette, showfliers=False)
plt.xlabel("F1-Score (Macro)") # Ajuste se for Binary ou Macro
plt.ylabel("") # Remove label redundante 'Modelo'
plt.tight_layout()
plt.savefig("fig_boxplot_f1.pdf", format='pdf', bbox_inches='tight')
plt.savefig("fig_boxplot_f1.png", dpi=300, bbox_inches='tight')
print("Gráfico 1 salvo: fig_boxplot_f1")

# ==============================================================================
# GRÁFICO 2: TRADE-OFF (EFICIÊNCIA vs EFICÁCIA)
# Scatter plot com erro padrão
# ==============================================================================
summary = df.groupby('Modelo').agg({
    'F1_Score': ['mean', 'std'],
    'Total_Latency_ms_per_1k': ['mean', 'std']
}).reset_index()

# Flatten columns
summary.columns = ['Modelo', 'F1_Mean', 'F1_Std', 'Lat_Mean', 'Lat_Std']

plt.figure(figsize=(10, 6))

for i, row in summary.iterrows():
    if row['Modelo'] == 'Dummy': continue # Ignorar Dummy neste gráfico pois distorce a escala
    
    # Define marcador diferente para o Ensemble destacar
    marker = '*' if 'Ensemble' in row['Modelo'] or 'Stacking' in row['Modelo'] else 'o'
    s_size = 300 if marker == '*' else 150
    
    plt.errorbar(
        row['Lat_Mean'], row['F1_Mean'],
        xerr=row['Lat_Std'], yerr=row['F1_Std'],
        fmt='none', ecolor='gray', alpha=0.5, capsize=3
    )
    plt.scatter(
        row['Lat_Mean'], row['F1_Mean'],
        label=row['Modelo'], color=model_colors[row['Modelo']],
        marker=marker, s=s_size, edgecolors='k', zorder=5
    )

plt.xlabel("Total Latency (ms / 1k samples)")
plt.ylabel("Mean F1-Score")
plt.legend(loc='lower right', frameon=True, fontsize=12)
plt.tight_layout()
plt.savefig("fig_scatter_tradeoff.pdf", format='pdf', bbox_inches='tight')
plt.savefig("fig_scatter_tradeoff.png", dpi=300, bbox_inches='tight')
print("Gráfico 2 salvo: fig_scatter_tradeoff")

# ==============================================================================
# GRÁFICO 3: TEMPO DE TREINAMENTO (BARPLOT LOG SCALE)
# Importante para justificar custo offline
# ==============================================================================
plt.figure(figsize=(10, 6))
# Ordenar por tempo de treino
order_time = df.groupby('Modelo')['Train_Time_sec'].mean().sort_values().index.tolist()

sns.barplot(
    x='Train_Time_sec', y='Modelo', data=df, 
    order=order_time, palette="rocket", errorbar='sd', capsize=.2
)

plt.xscale('log') # ESCALA LOGARÍTMICA É ESSENCIAL AQUI
plt.xlabel("Training Time (seconds) - Log Scale")
plt.ylabel("")
plt.tight_layout()
plt.savefig("fig_barplot_training_time.pdf", format='pdf', bbox_inches='tight')
plt.savefig("fig_barplot_training_time.png", dpi=300, bbox_inches='tight')
print("Gráfico 3 salvo: fig_barplot_training_time")