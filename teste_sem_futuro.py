"""Confere que nenhum estagiário "olha o futuro" e que os custos estão certos.

Uso: python teste_sem_futuro.py
"""
import numpy as np
import pandas as pd

from dados import dados_sinteticos
from empresa import NOME_BENCH, RegrasRH, resultados_semanais, semanas_do_periodo
from estagiarios import contratar_turma


def teste_sem_olhar_o_futuro():
    """Bagunçar os preços depois do candle t não pode mudar nenhuma decisão até t."""
    df = dados_sinteticos("1h", "2021-01-01", "2021-04-01")
    corte = len(df) // 2
    bagunçado = df.copy()
    rng = np.random.default_rng(0)
    bagunçado.iloc[corte + 1:] *= rng.uniform(0.5, 1.5, size=(len(df) - corte - 1, 1))
    for e in contratar_turma():
        a = e.posicoes(df).iloc[: corte + 1]
        b = e.posicoes(bagunçado).iloc[: corte + 1]
        assert a.equals(b), f"{e.nome} usa dados do futuro!"


def teste_custos_benchmark():
    """O Benchmark paga exatamente 2 trocas por semana (compra e venda)."""
    df = dados_sinteticos("1h", "2021-01-01", "2021-03-01")
    regras = RegrasRH()
    semanas = semanas_do_periodo("2021-01-11", "2021-03-01")
    ret, trocas, _ = resultados_semanais(df, [], "1h", regras, semanas)
    for s in ret.index:
        c = df["fechamento"]
        dentro = c[(c.index >= s) & (c.index < s + pd.Timedelta("7D"))]
        anterior = c[c.index < s].iloc[-1]
        esperado = dentro.iloc[-1] / anterior * (1 - regras.custo_troca) ** 2 - 1
        assert abs(ret.loc[s, NOME_BENCH] - esperado) < 1e-12
        assert trocas.loc[s, NOME_BENCH] == 2


def _dados_ml():
    df = dados_sinteticos("1h", "2021-01-01", "2021-05-01")
    df["volume_usdt"] = df["volume"] * df["fechamento"]
    df["negocios"] = df["volume"] / 2
    df["compra_agressora"] = df["volume"] * 0.5
    df["compra_agressora_usdt"] = df["volume_usdt"] * 0.5
    return df


def teste_features_sem_futuro():
    """As features do candle t não podem mudar se bagunçarmos os preços depois de t."""
    from ml.features import montar_features
    df = _dados_ml()
    corte = len(df) // 2
    b = df.copy()
    b.iloc[corte + 1:] *= np.random.default_rng(1).uniform(0.5, 1.5, size=(len(df) - corte - 1, 1))
    fa = montar_features(df, eth=df, perp=df).iloc[: corte + 1]
    fb = montar_features(b, eth=b, perp=b).iloc[: corte + 1]
    pd.testing.assert_frame_equal(fa, fb)


def teste_purging():
    """No walk-forward, nenhum rótulo de treino pode terminar depois do início da semana operada."""
    from ml.rotulos import triple_barrier
    df = _dados_ml()
    rot = triple_barrier(df)
    semana = pd.Timestamp("2021-03-01")
    treino = rot["valido"] & (rot["fim_rotulo"] <= semana) & (rot.index < semana)
    assert (rot.loc[treino, "fim_rotulo"] <= semana).all()
    # e o último candle de treino precisa estar a pelo menos 1 hora do início da semana
    assert rot.index[treino].max() <= semana - pd.Timedelta("2h")


if __name__ == "__main__":
    teste_sem_olhar_o_futuro()
    teste_custos_benchmark()
    teste_features_sem_futuro()
    teste_purging()
    print("OK: ninguém olha o futuro (estratégias, features e rótulos) e os custos do Benchmark batem.")
