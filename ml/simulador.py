"""Simulador de operações com barreiras (uma semana por vez).

- Começa a semana em dólar e termina em dólar.
- A decisão é tomada no fechamento de um candle e a entrada acontece nesse preço.
- Cada operação tem alvo, stop (as barreiras) e prazo máximo de H horas.
- Se o candle abre além da barreira (gap), a saída é no preço de abertura.
- Se alvo e stop caem no mesmo candle, conta o STOP (hipótese pessimista).
- Compras são no spot. Vendas (short) são no perpétuo e pagam/recebem funding a cada 8h.
- Custo por lado (taxa + spread/slippage) proporcional ao tamanho da posição.
"""
import numpy as np
import pandas as pd


def simular_semana(precos, decisoes, funding, custo_lado, horizonte=24):
    """precos: candles da semana (abertura, maxima, minima, fechamento).
    decisoes: DataFrame (mesmo índice) com lado, tamanho, largura.
    funding: Series do funding rate indexada pela hora do pagamento.
    Devolve (retorno da semana, lista de operações)."""
    o = precos["abertura"].to_numpy()
    h = precos["maxima"].to_numpy()
    l = precos["minima"].to_numpy()
    c = precos["fechamento"].to_numpy()
    lado = decisoes["lado"].to_numpy()
    tam = decisoes["tamanho"].to_numpy()
    larg = decisoes["largura"].to_numpy()
    fecha_em = precos.index + pd.Timedelta("1h")  # hora de fechamento de cada candle
    n = len(c)

    patrimonio = 1.0
    ops = []
    i = 0
    while i < n - 1:
        if lado[i] == 0 or tam[i] <= 0:
            i += 1
            continue
        s, f, entrada = lado[i], tam[i], c[i]
        sup, inf = entrada * np.exp(larg[i]), entrada * np.exp(-larg[i])
        saida, j, motivo = None, i, "tempo"
        for j in range(i + 1, min(i + horizonte, n - 1) + 1):
            stop = (l[j] <= inf) if s == 1 else (h[j] >= sup)
            alvo = (h[j] >= sup) if s == 1 else (l[j] <= inf)
            if stop:
                nivel = inf if s == 1 else sup
                saida = min(o[j], nivel) if s == 1 else max(o[j], nivel)
                motivo = "stop"
                break
            if alvo:
                nivel = sup if s == 1 else inf
                saida = max(o[j], nivel) if s == 1 else min(o[j], nivel)
                motivo = "alvo"
                break
        if saida is None:
            saida = c[j]
            motivo = "tempo" if j < n - 1 else "fim da semana"
        bruto = s * (saida / entrada - 1)
        receb_funding = 0.0
        if s == -1 and funding is not None:
            janela = funding[(funding.index > fecha_em[i]) & (funding.index <= fecha_em[j])]
            receb_funding = janela.sum()  # funding positivo: vendido recebe
        ret = f * (bruto + receb_funding) - 2 * f * custo_lado
        patrimonio *= 1 + ret
        ops.append({"entrada": fecha_em[i], "saida": fecha_em[j], "lado": int(s), "tamanho": f,
                    "retorno": ret, "motivo": motivo, "horas": j - i})
        i = j  # pode entrar de novo no fechamento do candle de saída
    return patrimonio - 1, ops
