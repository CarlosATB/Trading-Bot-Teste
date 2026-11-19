"""Coleta de dados: histórico de preços da Binance (API pública, sem conta).

Também gera dados sintéticos para testar o sistema sem internet.
Todas as datas estão em UTC. O índice de cada candle é a hora de ABERTURA dele.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

URL_KLINES = "https://api.binance.com/api/v3/klines"
URL_KLINES_FUT = "https://fapi.binance.com/fapi/v1/klines"
URL_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
URL_METRICAS = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
COLUNAS = ["data", "abertura", "maxima", "minima", "fechamento", "volume",
           "fechamento_ms", "volume_usdt", "negocios", "compra_agressora", "compra_agressora_usdt"]
PASTA_CACHE = Path(__file__).parent / "cache"
DURACAO = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def _get(url, params=None):
    import requests
    for tentativa in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if tentativa == 4:
                raise
            time.sleep(2 ** tentativa)


def baixar_binance(simbolo="BTCUSDT", intervalo="1h", inicio="2020-09-01", fim="2026-01-01",
                   mercado="spot"):
    """Baixa candles da Binance entre `inicio` e `fim` e guarda em CSV (cache).

    O cache só é aceito se cobrir o período pedido; senão baixa de novo.
    O último candle só entra se já estiver fechado.
    mercado="spot" (à vista) ou "futuros" (perpétuo USDT-M).
    Colunas extras: negocios (número de negócios) e compra_agressora (volume comprado a mercado).
    """
    PASTA_CACHE.mkdir(exist_ok=True)
    prefixo = "" if mercado == "spot" else "PERP_"
    arquivo = PASTA_CACHE / f"{prefixo}{simbolo}_{intervalo}_{inicio}_{fim}_v2.csv"
    url = URL_KLINES if mercado == "spot" else URL_KLINES_FUT
    limite = 1000 if mercado == "spot" else 1500
    if arquivo.exists():
        df = pd.read_csv(arquivo, index_col="data", parse_dates=True)
        if len(df):
            return df

    inicio_ms = int(pd.Timestamp(inicio, tz="UTC").timestamp() * 1000)
    fim_ms = int(pd.Timestamp(fim, tz="UTC").timestamp() * 1000) - 1
    agora_ms = int(time.time() * 1000)
    linhas = []
    while inicio_ms <= fim_ms:
        lote = _get(url, {"symbol": simbolo, "interval": intervalo,
                          "startTime": inicio_ms, "endTime": fim_ms, "limit": limite}).json()
        if not lote:
            break
        linhas.extend(lote)
        inicio_ms = lote[-1][0] + 1
        print(f"\r  baixando {prefixo}{simbolo} {intervalo}: {len(linhas):,} candles...", end="", flush=True)
        time.sleep(0.2)  # educação com a API
    print()

    linhas = [l for l in linhas if l[6] < agora_ms]  # só candles já fechados
    df = pd.DataFrame(linhas).iloc[:, :11]
    df.columns = COLUNAS
    df["data"] = pd.to_datetime(df["data"], unit="ms")
    df = df.set_index("data").drop(columns="fechamento_ms").astype(float)
    df = df[~df.index.duplicated()].sort_index()
    df.to_csv(arquivo)
    return df


def baixar_funding(simbolo="BTCUSDT", inicio="2019-09-01", fim="2026-01-01"):
    """Histórico do funding rate do perpétuo (pago a cada 8h). Positivo = comprados pagam vendidos."""
    PASTA_CACHE.mkdir(exist_ok=True)
    arquivo = PASTA_CACHE / f"FUNDING_{simbolo}_{inicio}_{fim}.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, index_col="data", parse_dates=True)["funding"]
    ini = int(pd.Timestamp(inicio, tz="UTC").timestamp() * 1000)
    fim_ms = int(pd.Timestamp(fim, tz="UTC").timestamp() * 1000) - 1
    linhas = []
    while ini <= fim_ms:
        lote = _get(URL_FUNDING, {"symbol": simbolo, "startTime": ini, "endTime": fim_ms, "limit": 1000}).json()
        if not lote:
            break
        linhas.extend(lote)
        ini = lote[-1]["fundingTime"] + 1
        time.sleep(0.2)
    s = pd.Series([float(l["fundingRate"]) for l in linhas],
                  index=pd.to_datetime([l["fundingTime"] for l in linhas], unit="ms"), name="funding")
    s.index = s.index.floor("min")  # alguns registros vêm com milissegundos de atraso
    s.index.name = "data"
    s = s[~s.index.duplicated()].sort_index()
    s.to_frame().to_csv(arquivo)
    return s


def baixar_metricas(simbolo="BTCUSDT", inicio="2020-09-01", fim="2026-01-01"):
    """Open interest e razões long/short dos futuros, a cada 5 minutos (data.binance.vision)."""
    import io
    import zipfile

    PASTA_CACHE.mkdir(exist_ok=True)
    arquivo = PASTA_CACHE / f"METRICAS_{simbolo}_{inicio}_{fim}.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, index_col="data", parse_dates=True)
    partes = []
    dias = pd.date_range(inicio, fim, freq="D", inclusive="left")
    for i, d in enumerate(dias):
        resp = _get(URL_METRICAS.format(s=simbolo, d=f"{d:%Y-%m-%d}"))
        if resp is not None:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                partes.append(pd.read_csv(z.open(z.namelist()[0])))
        print(f"\r  baixando métricas {simbolo}: {i + 1}/{len(dias)} dias...", end="", flush=True)
    print()
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["create_time"])
    df = df.set_index("data").drop(columns=["create_time", "symbol"]).apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={
        "sum_open_interest": "oi", "sum_open_interest_value": "oi_usdt",
        "count_toptrader_long_short_ratio": "top_ls_contas", "sum_toptrader_long_short_ratio": "top_ls_posicoes",
        "count_long_short_ratio": "ls_contas", "sum_taker_long_short_vol_ratio": "taker_ls",
    })
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
