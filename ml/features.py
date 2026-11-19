"""Features: o "resumo do mercado" que cada estagiário recebe a cada candle.

Regra de ouro: a linha do candle t só usa informação disponível no FECHAMENTO de t.
Todas as features são estacionárias (retornos, z-scores, razões), para o modelo
não depender do nível de preço do BTC, que mudou de 4 mil para 120 mil dólares.
"""
import numpy as np
import pandas as pd


def _z(s, janela):
    m = s.rolling(janela, min_periods=janela // 2).mean()
    d = s.rolling(janela, min_periods=janela // 2).std()
    return (s - m) / d.replace(0, np.nan)


def _rsi(c, n=14):
    delta = c.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 * ganho / (ganho + perda)


def volatilidade_horaria(df, span=168):
    """Volatilidade (desvio padrão do retorno log por candle), média exponencial de ~1 semana."""
    r = np.log(df["fechamento"]).diff()
    return r.ewm(span=span, min_periods=48).std()


def montar_features(spot, eth=None, perp=None, funding=None, metricas=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _montar(spot, eth, perp, funding, metricas)


def _montar(spot, eth, perp, funding, metricas):
    c = spot["fechamento"]
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=spot.index)

    # --- Preço e tendência (em várias escalas)
    for n in (1, 2, 4, 8, 24, 72, 168, 504):
        f[f"ret_{n}h"] = logc.diff(n)
    vol24 = r1.rolling(24).std()
    vol168 = r1.rolling(168).std()
    f["vol_24h"] = vol24
    f["vol_168h"] = vol168
    f["vol_razao"] = vol24 / vol168
    hl = np.log(spot["maxima"] / spot["minima"])
    f["parkinson_24h"] = np.sqrt((hl ** 2).rolling(24).mean() / (4 * np.log(2)))
    for n in (24, 72, 168, 500):
        f[f"dist_media_{n}h"] = (logc - logc.rolling(n).mean()) / (vol168 * np.sqrt(n))
    for n in (24, 168):
        mx, mn = spot["maxima"].rolling(n).max(), spot["minima"].rolling(n).min()
        f[f"posicao_faixa_{n}h"] = (c - mn) / (mx - mn)
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_14_168"] = _rsi(c, 14).rolling(168).mean()
    f["boll_z_20"] = _z(c, 20)
    f["assimetria_168h"] = r1.rolling(168).skew()
    # --- Tendência (v0.2: mais sensibilidade às altas)
    f["ret_1008h"] = logc.diff(1008)                       # 6 semanas
    f["dist_media_1000h"] = (logc - logc.rolling(1000).mean()) / (vol168 * np.sqrt(1000))
    for n in (24, 168):                                    # eficiência de Kaufman: 1 = tendência limpa
        f[f"eficiencia_{n}h"] = logc.diff(n).abs() / r1.abs().rolling(n).sum()
    f["horas_positivas_168h"] = (r1 > 0).rolling(168).mean() - 0.5
    f["queda_do_topo_168h"] = logc - np.log(spot["maxima"].rolling(168).max())
    f["alta_do_fundo_168h"] = logc - np.log(spot["minima"].rolling(168).min())
    t = np.arange(168) - 83.5
    f["inclinacao_168h"] = logc.rolling(168).apply(lambda y: np.dot(t, y) / np.dot(t, t), raw=True) / vol168
    f["media_72_acima_500"] = (logc.rolling(72).mean() - logc.rolling(500).mean()) / (vol168 * np.sqrt(500))
    f["tamanho_candle"] =(spot["fechamento"] - spot["abertura"]) / (spot["maxima"] - spot["minima"]).replace(0, np.nan)

    # --- Fluxo e microestrutura (vem nos próprios candles da Binance)
    agressao = spot["compra_agressora"] / spot["volume"].replace(0, np.nan)
    f["agressao_1h"] = agressao - 0.5
    f["agressao_24h"] = agressao.rolling(24).mean() - 0.5
    f["agressao_z"] = _z(agressao, 168)
    f["volume_z"] = _z(np.log1p(spot["volume_usdt"]), 168)
    f["negocios_z"] = _z(np.log1p(spot["negocios"]), 168)
    f["tamanho_medio_z"] = _z(np.log1p(spot["volume_usdt"] / spot["negocios"].replace(0, np.nan)), 168)
    f["fluxo_liquido_24h"] = ((2 * spot["compra_agressora_usdt"] - spot["volume_usdt"]).rolling(24).sum()
                              / spot["volume_usdt"].rolling(24).sum())

    # --- Calendário
    hora, dia = spot.index.hour, spot.index.dayofweek
    f["hora_sin"], f["hora_cos"] = np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24)
    f["dia_sin"], f["dia_cos"] = np.sin(2 * np.pi * dia / 7), np.cos(2 * np.pi * dia / 7)

    # --- ETH como ativo cruzado
    if eth is not None:
        le = np.log(eth["fechamento"].reindex(spot.index))
        for n in (4, 24, 168):
            f[f"eth_ret_{n}h"] = le.diff(n)
            f[f"eth_btc_{n}h"] = (le - logc).diff(n)
        f["correl_eth_168h"] = r1.rolling(168).corr(le.diff())

    # --- Derivativos: basis, funding e open interest
    if perp is not None:
        pc = perp["fechamento"].reindex(spot.index)
        basis = pc / c - 1
        f["basis"] = basis
        f["basis_z"] = _z(basis, 168)
        pagr = perp["compra_agressora"].reindex(spot.index) / perp["volume"].reindex(spot.index).replace(0, np.nan)
        f["perp_agressao_24h"] = pagr.rolling(24).mean() - 0.5
        f["perp_vs_spot_volume"] = np.log(perp["volume_usdt"].reindex(spot.index) / spot["volume_usdt"])
    if funding is not None:
        fu = funding.copy()
        fu.index = fu.index - pd.Timedelta("1h")  # o funding das 00:00 já é conhecido no fechamento do candle das 23:00
        fu = fu.reindex(spot.index.union(fu.index)).ffill().reindex(spot.index)
        f["funding"] = fu
        f["funding_3d"] = fu.rolling(72).mean()
        f["funding_z"] = _z(fu, 24 * 30)
    if metricas is not None:
        m = metricas.resample("1h", label="left", closed="left").last().reindex(spot.index)
        loi = np.log(m["oi_usdt"])
        f["oi_var_1h"] = loi.diff(1)
        f["oi_var_24h"] = loi.diff(24)
        f["oi_z"] = _z(loi, 168)
        f["top_ls_posicoes"] = np.log(m["top_ls_posicoes"])
        f["ls_contas"] = np.log(m["ls_contas"])
        f["taker_ls"] = np.log(m["taker_ls"])
        f["taker_ls_24h"] = np.log(m["taker_ls"]).rolling(24).mean()

    return f.replace([np.inf, -np.inf], np.nan).astype("float32")
