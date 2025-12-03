import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
import os

# ==============================================================================
# CONFIGURATION: ACM STYLE & ENGLISH
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "figure.figsize": (8, 6),
    "axes.grid": False
})

# FILE CONFIG
INPUT_FILE = "results/resultado_experimento_metrics.csv"
TARGET_MODEL_KEYWORD = "Stacking" # Keyword to identify your main model

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
if os.path.exists(INPUT_FILE):
    print(f"[INFO] Loading real data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    DATA_IS_REAL = True
else:
    print(f"[WARNING] File '{INPUT_FILE}' not found in {os.getcwd()}")
    print("[WARNING] Generating DUMMY data for demonstration purposes only!")
    DATA_IS_REAL = False
    rounds = []
    for i in range(1, 11):
        rounds.extend([
            {'Iteracao': i, 'Modelo': 'Stacking_Ensemble', 'F1_Score': 0.895 + np.random.normal(0, 0.005)},
            {'Iteracao': i, 'Modelo': 'CatBoost',          'F1_Score': 0.890 + np.random.normal(0, 0.005)},
            {'Iteracao': i, 'Modelo': 'LightGBM',          'F1_Score': 0.880 + np.random.normal(0, 0.005)},
            {'Iteracao': i, 'Modelo': 'XGBoost',           'F1_Score': 0.875 + np.random.normal(0, 0.005)},
            {'Iteracao': i, 'Modelo': 'RandomForest',      'F1_Score': 0.820 + np.random.normal(0, 0.01)},
            {'Iteracao': i, 'Modelo': 'KNN',               'F1_Score': 0.650 + np.random.normal(0, 0.02)}
        ])
    df = pd.DataFrame(rounds)

# Pivot Data (Rows=Rounds, Cols=Models)
pivot_df = df.pivot(index='Iteracao', columns='Modelo', values='F1_Score')

# Identify the "Champion" Model automatically
my_model = None
for col in pivot_df.columns:
    if TARGET_MODEL_KEYWORD in col:
        my_model = col
        break

if my_model is None:
    # Fallback if keyword not found
    my_model = pivot_df.columns[0]
    print(f"[WARN] Keyword '{TARGET_MODEL_KEYWORD}' not found. Using '{my_model}' as baseline.")
else:
    print(f"[INFO] Main Model identified as: {my_model}")

# ==============================================================================
# 2. FRIEDMAN TEST (Global Difference)
# ==============================================================================
data_matrix = [pivot_df[col].values for col in pivot_df.columns]
stat, p_friedman = friedmanchisquare(*data_matrix)

print("\n" + "="*60)
print(f"FRIEDMAN TEST (Global Omnibus)")
print(f"Statistic: {stat:.4f}, P-value: {p_friedman:.4e}")
if p_friedman < 0.05:
    print(">> CONCLUSION: There is a statistically significant difference among models.")
else:
    print(">> CONCLUSION: No significant difference found globally.")
print("="*60)

# ==============================================================================
# 3. WILCOXON SIGNED-RANK TEST (Pairwise: Main vs Others)
# ==============================================================================
results_wilcoxon = []
comparison_models = [m for m in pivot_df.columns if m != my_model]

print(f"\nWILCOXON TEST ({my_model} vs ...)")
print(f"{'Opponent':<20} | {'P-Value':<12} | {'Significant?':<10}")
print("-" * 60)

for opponent in comparison_models:
    # alternative='greater' tests if my_model > opponent
    stat, p_val = wilcoxon(pivot_df[my_model], pivot_df[opponent], alternative='greater')
    
    is_sig = "YES" if p_val < 0.05 else "No"
    results_wilcoxon.append({
        'Opponent': opponent,
        'P_Value': p_val,
        'Significant': p_val < 0.05
    })
    print(f"{opponent:<20} | {p_val:.6f}     | {is_sig}")

# ==============================================================================
# 4. GENERATE LATEX TABLE (ACM READY)
# ==============================================================================
print("\n--- LaTeX Table Snippet ---")
latex_code = r"""
\begin{table}[ht]
\centering
\caption{Statistical comparison using Wilcoxon Signed-Rank Test (one-tailed) between the proposed Stacking Ensemble and base models across 10 folds ($N=10$, $\alpha=0.05$).}
\label{tab:wilcoxon}
\begin{tabular}{lcc}
\toprule
\textbf{Comparison} & \textbf{P-value} & \textbf{Significant?} \\
\midrule
"""
for res in results_wilcoxon:
    # Format P-value (< 0.001 convention)
    if res['P_Value'] < 0.001:
        p_str = "$< 0.001$"
    else:
        p_str = f"{res['P_Value']:.4f}"
        
    sig_str = r"\textbf{Yes}" if res['Significant'] else "No"
    
    # Clean model names for LaTeX (Replace underscores)
    opp_name = res['Opponent'].replace('_', ' ')
    my_name_clean = "Stacking" # Short name for the table
    
    latex_code += f"{my_name_clean} vs. {opp_name} & {p_str} & {sig_str} \\\\\n"

latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""
print(latex_code)

# ==============================================================================
# 5. VISUALIZATION: P-VALUE HEATMAP
# ==============================================================================
models = pivot_df.columns
n_models = len(models)
p_matrix = np.ones((n_models, n_models))

for i in range(n_models):
    for j in range(n_models):
        if i == j: continue
        try:
            # Test: Row Model > Col Model
            _, p = wilcoxon(pivot_df.iloc[:, i], pivot_df.iloc[:, j], alternative='greater')
            p_matrix[i, j] = p
        except ValueError:
            p_matrix[i, j] = 1.0

plt.figure(figsize=(10, 8))
mask = np.eye(n_models, dtype=bool)

ax = sns.heatmap(
    p_matrix, 
    xticklabels=[m.replace('_', '\n') for m in models], 
    yticklabels=[m.replace('_', ' ') for m in models],
    annot=True, 
    fmt=".3f", 
    cmap="Greens_r", # Dark Green = Significant (Low p-value)
    cbar_kws={'label': 'P-value (Wilcoxon Greater)'},
    mask=mask,
    vmin=0, vmax=0.05 # Focus contrast on significant range
)

plt.title("Statistical Significance Heatmap ($H_1$: Row Model $>$ Col Model)")
plt.xlabel("Opponent Model")
plt.ylabel("Reference Model")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

filename = "fig_significance_heatmap.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n[INFO] Heatmap saved to: {filename}")

filename = "fig_significance_heatmap.pdf"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n[INFO] Heatmap saved to: {filename}")