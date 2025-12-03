#!/usr/bin/env python3
import requests
from pathlib import Path

# ============================
# CONFIGURAÇÕES
# ============================

# Netdata local
BASE_URL = "http://127.0.0.1:19999"

# Intervalo de interesse (em segundos Unix / epoch)
# 01/12/2025 18:55:29 GMT-3
TRAIN_START_TIME = 1764626129

# 03/12/2025 12:02:45 GMT-3
TRAIN_END_TIME = 1764774165

# Quantidade de pontos que você quer que o Netdata retorne
# (Netdata vai agregar os dados para caber nesse número de pontos)
# Se quiser aproximar 1 ponto por segundo, use a diferença:
#   1764774165 - 1764626129 = 148036
# Mas 148k pontos pode ficar pesado; 3600 costuma ser suficiente.
POINTS = 148036

# Pasta de saída
OUTPUT_DIR = Path("netdata_csv")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_url(chart: str) -> str:
    """Monta a URL da API do Netdata para um chart específico."""
    return (
        f"{BASE_URL}/api/v1/data?"
        f"chart={chart}"
        f"&after={TRAIN_START_TIME}"
        f"&before={TRAIN_END_TIME}"
        f"&points={POINTS}"
        f"&format=csv"
    )


def fetch_and_save(chart: str, filename: str):
    """Baixa o CSV de um chart e salva em disco."""
    url = build_url(chart)
    print(f"Baixando {chart} de {url}")

    try:

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERRO] Falha ao coletar {chart}: {e}")
        return

    out_path = OUTPUT_DIR / filename
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"[OK] {chart} salvo em: {out_path}")


def main():
    # CPU total
    fetch_and_save("system.cpu", "cpu.csv")

    fetch_and_save("system.ctxt", "ctx_switch.csv")

    #system.load
    fetch_and_save("system.load", "load_average.csv")

    #system.cpu_some_pressure
    fetch_and_save("system.cpu_some_pressure", "cpu_pressure.csv")

    #system.interrupts
    fetch_and_save("system.interrupts", "interrupts.csv")

    #system.ram
    fetch_and_save("system.ram", "ram.csv")

    # RAM disponível
    fetch_and_save("mem.available", "ram_available.csv")

    # RAM comprometida/uso
    fetch_and_save("mem.committed", "ram_used.csv")


if __name__ == "__main__":
    main()
