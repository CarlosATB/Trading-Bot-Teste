"""Walk-forward semana a semana: o estagiário só aprende com o passado e se atualiza toda semana.

Estudo inicial (uma vez só, antes de começar): o estagiário aprende com todo o histórico
disponível (2019 em diante) e passa a primeira semana só observando, sem operar.
Depois disso ele opera toda semana. No fim de cada semana ele se atualiza:
  1. Erros: os candles em que errou ao vivo ganham mais peso no treino seguinte.
  2. Treino: todo o histórico com rótulo já encerrado antes da próxima semana começar (purging).
     Um rótulo que termina dentro da semana operada olharia o futuro, então é descartado.
  3. Pesos: (a) unicidade do rótulo (rótulos sobrepostos contam menos, López de Prado),
            (b) recência (meia-vida de 1 ano), (c) reforço nos erros.
  4. Previsão: probabilidades para cada candle da semana, usando só features até aquele candle.
"""
import time

import numpy as np
import pandas as pd

from ml.features import volatilidade_horaria
from ml.regra import RegraParaTodos, decidir
from ml.simulador import simular_semana

UMA_SEMANA = pd.Timedelta("7D")


def unicidade(rot):
    """Peso de unicidade média de cada rótulo (rótulos que se sobrepõem dividem o crédito)."""
    n = len(rot)
    t = rot["tempo"].to_numpy().astype(int)
    ini = np.arange(n) + 1
    fim = np.minimum(np.arange(n) + t, n - 1)
    conc = np.zeros(n + 2)
    np.add.at(conc, ini, 1)
    np.add.at(conc, fim + 1, -1)
    conc = np.cumsum(conc)[:n]
    inv = np.where(conc > 0, 1 / np.maximum(conc, 1), 0)
    acum = np.concatenate([[0], np.cumsum(inv)])
    u = (acum[fim + 1] - acum[ini]) / np.maximum(fim - ini + 1, 1)
    return pd.Series(u, index=rot.index)


def contexto_da_regra(rot, feats, spot, semana, treino):
    """Tudo o que a regra precisa além das probabilidades (só com dados do passado)."""
    idx = rot.index[(rot.index >= semana) & (rot.index < semana + UMA_SEMANA)]
    ctx = pd.DataFrame(index=idx)
    ctx["largura"] = rot.loc[idx, "largura"]
    r_tr = rot.loc[treino]
    ctx["ret_tempo"] = np.exp(r_tr.loc[r_tr["rotulo"] == 0, "ret_saida"]).mean() - 1
    for nome, k in (("t_alta", 1), ("t_baixa", -1), ("t_tempo", 0)):
        ctx[nome] = r_tr.loc[r_tr["rotulo"] == k, "tempo"].mean()
    ctx["funding"] = feats.loc[idx, "funding"].fillna(0.0) if "funding" in feats else 0.0
    ctx["vol_anual"] = volatilidade_horaria(spot).loc[idx] * np.sqrt(24 * 365)
    ctx["horas_restantes"] = ((semana + UMA_SEMANA) - (idx + pd.Timedelta("1h"))) / pd.Timedelta("1h")
    return ctx


def prever_walkforward(estagiario, feats, rot, semanas, meia_vida_dias=365, reforco_erro=1.0,
                       inicio_treino=None, log=print):
    """Gera as probabilidades fora da amostra, semana a semana. Devolve DataFrame com p_baixa, p_tempo, p_alta."""
    X_all = feats.to_numpy()
    y_all = rot["rotulo"].to_numpy()
    u_all = unicidade(rot).to_numpy()
    valido = rot["valido"].to_numpy()
    if inicio_treino is not None:
        valido = valido & (rot.index >= pd.Timestamp(inicio_treino))
    fim_rot = rot["fim_rotulo"].to_numpy()
    reforco = np.ones(len(rot))
    previsoes = []
    t0 = time.time()
    for k, semana in enumerate(semanas):
        treino = valido & (fim_rot <= np.datetime64(semana)) & (rot.index < semana)
        idade = (semana - rot.index[treino]).days.to_numpy()
        peso = u_all[treino] * 0.5 ** (idade / meia_vida_dias) * reforco[treino]
        estagiario.treinar(X_all[treino], y_all[treino], peso / peso.mean())

        na_semana = (rot.index >= semana) & (rot.index < semana + UMA_SEMANA)
        p = estagiario.prever(X_all[na_semana])
        prev = pd.DataFrame(p, index=rot.index[na_semana], columns=["p_baixa", "p_tempo", "p_alta"])
        previsoes.append(prev)

        # estudo dos erros desta semana (usado quando os rótulos dela já tiverem terminado)
        verdade = y_all[na_semana] + 1
        p_certo = p[np.arange(len(p)), verdade]
        reforco[na_semana] = 1 + reforco_erro * (1 - p_certo)

        if (k + 1) % 20 == 0 or k == len(semanas) - 1:
            log(f"    {estagiario.nome}: {k + 1}/{len(semanas)} semanas ({time.time() - t0:.0f}s)")
    return pd.concat(previsoes)


def operar(estagiario, previsoes, feats, rot, spot, funding, semanas, regra=RegraParaTodos(),
           horizonte=24, semana_de_estudo=True, historico_meta=None):
    """Aplica a regra para todos às previsões e simula cada semana.

    A Luna, além disso, usa o meta-modelo treinado com as próprias entradas passadas.
    A primeira semana é o estudo inicial (só observa, não opera)."""
    usa_meta = hasattr(estagiario, "treinar_meta")
    custo_lado = regra.taxa + regra.spread_slippage
    historico_meta = [] if historico_meta is None else historico_meta
    ret, ops_todas, linhas_dec = {}, [], []
    for k, semana in enumerate(semanas):
        idx = previsoes.index[(previsoes.index >= semana) & (previsoes.index < semana + UMA_SEMANA)]
        if len(idx) < 160:
            continue
        treino = rot["valido"] & (rot["fim_rotulo"] <= semana) & (rot.index < semana)
        ctx = contexto_da_regra(rot, feats, spot, semana, treino)
        dec = decidir(previsoes.loc[idx], ctx, regra)
        dec["largura"] = ctx["largura"]

        if usa_meta:
            X_meta = pd.concat([feats.loc[idx], previsoes.loc[idx], dec[["ev", "tempo_esperado"]]], axis=1)
            dec["lado_primario"] = dec["lado"]
            passado = [h for h in historico_meta if h.index.max() < semana]
            if passado:
                hist = pd.concat(passado)
                hist = hist[rot.loc[hist.index, "fim_rotulo"] <= semana]
                estagiario.treinar_meta(hist.drop(columns="acertou"), hist["acertou"])
            q = estagiario.confianca_meta(X_meta)
            dec["confianca_meta"] = q
            vetar = q < 0.5
            dec.loc[vetar, "lado"] = 0
            dec["tamanho"] = np.where(vetar, 0.0, dec["tamanho"] * q)
            # guarda as propostas desta semana para o estudo futuro (deu lucro ou não?)
            propostas = dec["lado_primario"] != 0
            if propostas.any():
                r = rot.loc[idx[propostas]]
                lucro = dec.loc[propostas, "lado_primario"] * (np.exp(r["ret_saida"]) - 1) - regra.custo_ida_volta
                h = X_meta.loc[propostas].copy()
                h["acertou"] = (lucro > 0).astype(int)
                historico_meta.append(h[r["valido"]])

        if k == 0 and semana_de_estudo:
            dec["lado"], dec["tamanho"] = 0, 0.0  # estudo inicial: só observa
        precos = spot.loc[idx, ["abertura", "maxima", "minima", "fechamento"]]
        r_sem, ops = simular_semana(precos, dec, funding, custo_lado, horizonte)
        ret[semana] = r_sem
        for o in ops:
            o["semana"] = semana
        ops_todas += ops
        linhas_dec.append(dec)
    return (pd.Series(ret, name=estagiario.nome), pd.DataFrame(ops_todas), pd.concat(linhas_dec),
            historico_meta)
