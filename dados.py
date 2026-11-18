"""Coleta de dados: histórico de preços da Binance (API pública, sem conta).

Também gera dados sintéticos para testar o sistema sem internet.
Todas as datas estão em UTC. O índice de cada candle é a hora de ABERTURA dele.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

URL_KLINES = "https://api.binance.com/api/v3/klines"
PASTA_CACHE = Path(__file__).parent / "cache"
DURACAO = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def baixar_binance(simbolo="BTCUSDT", intervalo="1h", inicio="2020-09-01", fim="2026-01-01"):
    """Baixa candles da Binance entre `inicio` e `fim` e guarda em CSV (cache).

    O cache só é aceito se cobrir o período pedido; senão baixa de novo.
    O último candle só entra se já estiver fechado.
    """
    import requests

    PASTA_CACHE.mkdir(exist_ok=True)
    arquivo = PASTA_CACHE / f"{simbolo}_{intervalo}_{inicio}_{fim}.csv"
    if arquivo.exists():
        df = pd.read_csv(arquivo, index_col="data", parse_dates=True)
        if len(df):
            return df

    inicio_ms = int(pd.Timestamp(inicio, tz="UTC").timestamp() * 1000)
    fim_ms = int(pd.Timestamp(fim, tz="UTC").timestamp() * 1000) - 1
    agora_ms = int(time.time() * 1000)
    linhas = []
    while inicio_ms <= fim_ms:
        for tentativa in range(5):
            try:
                resp = requests.get(URL_KLINES, params={
                    "symbol": simbolo, "interval": intervalo,
                    "startTime": inicio_ms, "endTime": fim_ms, "limit": 1000,
                }, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if tentativa == 4:
                    raise
                time.sleep(2 ** tentativa)
        lote = resp.json()
        if not lote:
            break
        linhas.extend(lote)
        inicio_ms = lote[-1][0] + 1
        print(f"\r  baixando {simbolo} {intervalo}: {len(linhas):,} candles...", end="", flush=True)
        time.sleep(0.2)  # educação com a API
    print()

    linhas = [l for l in linhas if l[6] < agora_ms]  # só candles já fechados
    df = pd.DataFrame(linhas).iloc[:, :6]
    df.columns = ["data", "abertura", "maxima", "minima", "fechamento", "volume"]
    df["data"] = pd.to_datetime(df["data"], unit="ms")
    df = df.set_index("data").astype(float)
    df = df[~df.index.duplicated()].sort_index()
    df.to_csv(arquivo)
    return df


def relatorio_buracos(df, intervalo):
    """Lista os buracos nos dados (a Binance teve algumas horas fora do ar)."""
    passo = pd.Timedelta(DURACAO[intervalo])
    dif = df.index.to_series().diff()
    buracos = dif[dif > passo]
    faltando = int(((buracos / passo) - 1).sum())
    return faltando, buracos


def dados_sinteticos(intervalo="1h", inicio="2020-09-01", fim="2026-01-01", semente=42):
    """Preços falsos com fases de alta, queda e lateralização (só para testes)."""
    rng = np.random.default_rng(semente)
    datas = pd.date_range(inicio, fim, freq=DURACAO[intervalo], inclusive="left", name="data")
    n = len(datas)
    por_dia = pd.Timedelta("1D") / pd.Timedelta(DURACAO[intervalo])
    fase = int(60 * por_dia)  # cada fase dura ~60 dias
    tendencias = rng.choice([0.002, -0.0015, 0.0], size=n // fase + 1) / por_dia
    drift = np.repeat(tendencias, fase)[:n]
    retornos = drift + rng.normal(0, 0.035 / np.sqrt(por_dia), n)
    fechamento = 10_000 * np.exp(np.cumsum(retornos))
    abertura = np.concatenate([[fechamento[0]], fechamento[:-1]])
    ruido = np.abs(rng.normal(0, 0.01 / np.sqrt(por_dia), n))
    return pd.DataFrame({
        "abertura": abertura,
        "maxima": np.maximum(abertura, fechamento) * (1 + ruido),
        "minima": np.minimum(abertura, fechamento) * (1 - ruido),
        "fechamento": fechamento,
        "volume": rng.uniform(1e3, 1e4, n),
    }, index=datas)
