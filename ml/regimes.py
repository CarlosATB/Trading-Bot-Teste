"""Avaliação por regime de mercado (proposta do Carlos, v0.2).

Cada semana é classificada pelo retorno do BTC (o Benchmark) naquela semana:

| Regime          | Retorno do BTC na semana | O que avaliar                          |
|-----------------|--------------------------|----------------------------------------|
| Alta forte      | >= +5%                   | Captura da valorização do BTC          |
| Alta moderada   | +1,5% a +5%              | Capacidade de acompanhar tendências    |
| Lateral         | -1,5% a +1,5%            | Evitar operações desnecessárias        |
| Queda moderada  | -5% a -1,5%              | Proteção do capital                    |
| Queda forte     | <= -5%                   | Aproveitar oportunidades de short      |

A classificação usa o resultado da própria semana, então serve só para AVALIAR (depois que a
semana acabou). Nenhum estagiário vê o regime da semana em que está operando.

Métricas:
- Capture Ratio (alta): retorno médio do bot nas semanas de alta / retorno médio do BTC nelas.
  100% = acompanhou a alta inteira; 0% = ficou parado.
- Downside Capture (queda): retorno médio do bot nas semanas de queda / retorno médio do BTC nelas.
  100% = caiu junto; 0% = evitou a queda; negativo = ganhou dinheiro na queda (short).
"""
import numpy as np
import pandas as pd

REGIMES = ["Alta forte", "Alta moderada", "Lateral", "Queda moderada", "Queda forte"]
LIMITES = [-np.inf, -0.05, -0.015, 0.015, 0.05, np.inf]
ROTULOS_CORTE = ["Queda forte", "Queda moderada", "Lateral", "Alta moderada", "Alta forte"]


def classificar(ret_bench):
    """Regime de cada semana a partir do retorno do Benchmark."""
    return pd.cut(ret_bench, LIMITES, labels=ROTULOS_CORTE, right=False).astype(str)


def capture_ratio(r, bench):
    alta = bench > 0
    return r[alta].mean() / bench[alta].mean() if alta.any() else np.nan


def downside_capture(r, bench):
    queda = bench < 0
    return r[queda].mean() / bench[queda].mean() if queda.any() else np.nan


def tabela_por_regime(ret, bench, ops=None):
    """Para cada bot e regime: semanas, retorno médio, % semanas ganhando do BTC e operações/semana.

    ret: DataFrame (semanas x bots). bench: Series. ops: dict nome -> DataFrame de operações (opcional).
    """
    regime = classificar(bench)
    linhas = []
    for nome in ret.columns:
        r = ret[nome]
        o = ops.get(nome) if ops else None
        for reg in REGIMES:
            m = regime == reg
            if not m.any():
                continue
            linha = {"Bot": nome, "Regime": reg, "Semanas": int(m.sum()),
                     "Retorno médio": r[m].mean(), "BTC médio": bench[m].mean(),
                     "Ganhou do BTC": (r[m] > bench[m]).mean()}
            if o is not None and len(o):
                sem = set(r.index[m])
                om = o[o["semana"].isin(sem)]
                linha["Operações/semana"] = len(om) / m.sum()
                linha["% vendas (short)"] = (om["lado"] == -1).mean() if len(om) else np.nan
            linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_capture(ret, bench):
    return pd.DataFrame({
        nome: {"Capture Ratio (altas)": capture_ratio(ret[nome], bench),
               "Downside Capture (quedas)": downside_capture(ret[nome], bench)}
        for nome in ret.columns}).T
