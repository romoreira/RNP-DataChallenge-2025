#!/usr/bin/env python3
"""
Analisa métricas do Netdata e gera figuras no estilo ACM (Serif, PDF, High Contrast).
Sem títulos nos gráficos (deixar para o caption do LaTeX).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# CONFIGURAÇÕES GLOBAIS & ESTILO ACM
# ============================

METRICS_DIR = Path("netdata_metrics/netdata_csv")
FIG_DIR = Path("graphs/figs")
FIG_DIR.mkdir(exist_ok=True)

EXPERIMENT_LOG = Path("results/resultado_experimento_timestamp.csv")

# Configuração de Estilo para Publicação (ACM)
def set_acm_style():
    plt.rcParams.update({
        # Fontes
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"], # Tenta Times, fallback para DejaVu
        "font.size": 16,
        
        # Tamanhos específicos
        "axes.labelsize": 16,
        # "axes.titlesize": 18, # Removido para evitar tentação de usar títulos
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        
        # Linhas e Marcadores
        "lines.linewidth": 2,
        "lines.markersize": 8,
        
        # Layout
        "figure.autolayout": True, # Ajusta layout automaticamente
        "savefig.format": "pdf",   # Padrão vetorial
        "savefig.bbox": "tight",   # Remove bordas brancas excessivas
    })

# ============================
# CARREGAR DADOS (Lógica Original Mantida)
# ============================

def load_metrics():
    # Assumindo que a lógica de carga está correta e inalterada
    try:
        cpu = pd.read_csv(METRICS_DIR / "cpu.csv", parse_dates=["time"]).sort_values("time")
        cpu_cols = [c for c in cpu.columns if c in ["user", "system", "nice", "softirq", "irq", "steal"]]
        cpu["cpu_active"] = cpu[cpu_cols].sum(axis=1)

        load_avg = pd.read_csv(METRICS_DIR / "load_average.csv", parse_dates=["time"]).sort_values("time")
        ctxt = pd.read_csv(METRICS_DIR / "ctx_switch.csv", parse_dates=["time"]).sort_values("time")
        ram = pd.read_csv(METRICS_DIR / "ram.csv", parse_dates=["time"]).sort_values("time")
        ram_avail = pd.read_csv(METRICS_DIR / "ram_available.csv", parse_dates=["time"]).sort_values("time")
        
        ram_used_path = METRICS_DIR / "ram_used.csv"
        ram_committed = None
        if ram_used_path.exists():
            ram_committed = pd.read_csv(ram_used_path, parse_dates=["time"]).sort_values("time")

        metrics = cpu.copy()
        for df in [load_avg, ctxt, ram, ram_avail]:
            metrics = pd.merge_asof(metrics.sort_values("time"), df.sort_values("time"), on="time", direction="nearest")
        
        if ram_committed is not None:
            metrics = pd.merge_asof(metrics.sort_values("time"), ram_committed.sort_values("time"), on="time", direction="nearest")
            
        return metrics.sort_values("time")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique o caminho: {e}")
        return pd.DataFrame()

def load_events():
    if not EXPERIMENT_LOG.exists():
        print(f"Erro: Log de experimento não encontrado em {EXPERIMENT_LOG}")
        return pd.DataFrame()
        
    events = pd.read_csv(EXPERIMENT_LOG, parse_dates=["Start_Time", "End_Time"])
    return events.sort_values("Start_Time")

def summarise_event_resources(metrics, event_row):
    t0 = event_row["Start_Time"]
    t1 = event_row["End_Time"]
    mask = (metrics["time"] >= t0) & (metrics["time"] <= t1)
    sub = metrics.loc[mask]

    if sub.empty:
        return {"cpu_active_mean": np.nan, "ram_used_mean": np.nan}

    cpu_mean = sub["cpu_active"].mean()
    ram_mean = sub["used"].mean() if "used" in sub.columns else np.nan
    
    return {"cpu_active_mean": cpu_mean, "ram_used_mean": ram_mean}

def build_event_resource_summary(metrics, events):
    rows = []
    for _, ev in events.iterrows():
        stats = summarise_event_resources(metrics, ev)
        row = ev.to_dict()
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows)

# ============================
# FIGURAS REMODELADAS (Sem Títulos)
# ============================

def plot_resource_by_model(summary):
    """
    Figura 2: Barras por modelo.
    Melhorias: Cores contrastantes, hachuras (patterns) e legenda externa.
    """
    train = summary[summary["Stage"] == "Training"].copy()
    if train.empty: return

    agg = train.groupby("Model").agg(
        duration_mean=("Duration_Sec", "mean"),
        cpu_active_mean=("cpu_active_mean", "mean")
    ).reset_index()

    models = agg["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Eixo 1: Tempo (Barras Cinza Escuro Sólido)
    bar1 = ax1.bar(
        x - width / 2, 
        agg["duration_mean"], 
        width, 
        color='#404040', # Cinza escuro
        label="Training Time (s)",
        edgecolor='black'
    )
    ax1.set_ylabel("Training Time (s)", color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.tick_params(axis='y', labelcolor='black')

    # Eixo 2: CPU (Barras Brancas com Hachuras)
    ax2 = ax1.twinx()
    bar2 = ax2.bar(
        x + width / 2, 
        agg["cpu_active_mean"], 
        width, 
        color='white',
        edgecolor='black',
        hatch='///', # Padrão listrado (bom para papers P&B)
        label="Mean CPU Active (%)"
    )
    ax2.set_ylabel("Mean CPU Active (%)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Limites para dar respiro vertical
    ax1.set_ylim(0, agg["duration_mean"].max() * 1.2)
    ax2.set_ylim(0, 100) # CPU é porcentagem, fixar ou dar margem

    # Legenda Unificada no topo (sem título da legenda também, se preferir)
    lines = [bar1, bar2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=2, frameon=False) 

    ax1.set_xlabel("Model")
    
    # NENHUM plt.title() AQUI
    
    output_path = FIG_DIR / "fig_resource_by_model.pdf"
    fig.savefig(output_path, dpi=300)
    print(f"Gerado: {output_path}")
    plt.close(fig)

def plot_duration_vs_cpu(summary):
    """
    Figura 3: Dispersão Log-Scale.
    Melhorias: Marcadores maiores, grade de leitura, estilo limpo.
    """
    train = summary[summary["Stage"] == "Training"].copy()
    if train.empty: return
    
    train = train[train["Duration_Sec"] > 0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Definir marcadores diferentes para garantir distinção se impresso em P&B
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    unique_models = train["Model"].unique()
    
    for i, model in enumerate(unique_models):
        sub = train[train["Model"] == model]
        marker = markers[i % len(markers)]
        
        ax.scatter(
            sub["Duration_Sec"],
            sub["cpu_active_mean"],
            label=model,
            s=120,          # Tamanho grande para visibilidade
            alpha=0.75,     
            edgecolors="black", # Borda preta nítida
            linewidths=1.0,
            marker=marker
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Duration (s, log scale)")
    ax.set_ylabel("Mean CPU Active (%)")
    
    # Grade suave para facilitar leitura logarítmica
    ax.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax.grid(True, which="minor", linestyle=":", alpha=0.2, color='gray')

    # Legenda simplificada (mantendo o título da LEGENDA "Models" para contexto, 
    # mas o gráfico em si não tem título)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Models", frameon=False)

    # NENHUM plt.title() AQUI

    output_path = FIG_DIR / "fig_duration_vs_cpu.pdf"
    fig.savefig(output_path, dpi=300)
    print(f"Gerado: {output_path}")
    plt.close(fig)

# Função de timeline removida do fluxo principal conforme pedido anterior, 
# mas mantida no código caso precise futuramente.
def plot_cpu_ram_timeline(metrics, events):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(metrics["time"], metrics["cpu_active"], color='black', linewidth=1.5, label="CPU Active (%)")
    ax1.set_ylabel("CPU Active (%)")
    ax2 = ax1.twinx()
    if "used" in metrics.columns:
        ax2.plot(metrics["time"], metrics["used"], color='gray', linestyle="--", linewidth=1.5, label="RAM Used")
        ax2.set_ylabel("RAM Used (MiB)")
    for _, ev in events.iterrows():
        if str(ev["Stage"]).lower() == "training":
            ax1.axvspan(ev["Start_Time"], ev["End_Time"], color='gray', alpha=0.2)
        elif str(ev["Stage"]).lower() == "inference":
            ax1.axvspan(ev["Start_Time"], ev["End_Time"], color='gray', alpha=0.4)
    fig.autofmt_xdate()
    ax1.set_xlabel("Time")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=1)
    # NENHUM plt.title() AQUI
    output_path = FIG_DIR / "fig_cpu_ram_timeline.pdf"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# ============================
# MAIN
# ============================

def main():
    set_acm_style() # Aplica o estilo globalmente
    
    print("Carregando dados...")
    metrics = load_metrics()
    events = load_events()
    
    if metrics.empty or events.empty:
        print("Dados insuficientes para gerar gráficos.")
        return

    print("Processando resumos...")
    summary = build_event_resource_summary(metrics, events)
    summary.to_csv("event_resource_summary.csv", index=False)

    print("Gerando figuras (PDF)...")
    plot_resource_by_model(summary)
    plot_duration_vs_cpu(summary)
    # plot_cpu_ram_timeline(metrics, events) # Removido conforme solicitado

    print("Concluído.")

if __name__ == "__main__":
    main()