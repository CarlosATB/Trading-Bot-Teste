"""Empresa de Trading, fase 1 de Machine Learning (candles de 1h, BTC/USDT).

Uso:
    python main_ml.py                  # treino walk-forward 2020-2023 (2020 = estágio, RH a partir de 2021)
    python main_ml.py --prova          # também roda a prova das 20 semanas sorteadas (2024-2025)
    python otimizar.py                 # (antes) Optuna escolhe os hiperparâmetros só com dados até 2023

A prova deve ser olhada o MÍNIMO de vezes. Cada vez que ela roda fica registrada em
relatorios/registro_da_prova.csv, e isso entra na conta do Deflated Sharpe.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dados import PASTA_CACHE, baixar_binance, baixar_funding, baixar_metricas
from empresa import NOME_BENCH, RegrasRH, avaliar_prova, queda_maxima, resultados_semanais, \
    rodar_empresa, semanas_do_periodo
from ml.estatistica import dsr, sharpe
from ml.features import montar_features
from ml.modelos import contratar_turma_ml
from ml.regra import RegraParaTodos
from ml.rotulos import triple_barrier
from ml.walkforward import operar, prever_walkforward

RAIZ = Path(__file__).parent
REL = RAIZ / "relatorios"
CACHE_ML = PASTA_CACHE / "ml"
INICIO_DADOS = "2019-01-01"
FIM_DADOS = "2026-01-01"
ESTAGIO = ("2020-01-06", "2021-01-04")   # aprendizado inicial (não conta para o RH)
TREINO = ("2021-01-04", "2024-01-01")    # avaliado pelo RH
HORIZONTE = 24                            # barreira de tempo: 24 horas
K_BARREIRA = 1.0                          # largura = 1 desvio padrão esperado em 24h


def carregar():
    spot = baixar_binance("BTCUSDT", "1h", INICIO_DADOS, FIM_DADOS)
    eth = baixar_binance("ETHUSDT", "1h", INICIO_DADOS, FIM_DADOS)
    perp = baixar_binance("BTCUSDT", "1h", "2019-09-01", FIM_DADOS, mercado="futuros")
    funding = baixar_funding("BTCUSDT", "2019-09-01", FIM_DADOS)
    metricas = baixar_metricas("BTCUSDT", "2020-06-01", FIM_DADOS)
    feats = montar_features(spot, eth, perp, funding, metricas)
    rot = triple_barrier(spot, HORIZONTE, K_BARREIRA)
    return spot, funding, feats, rot


def previsoes_com_cache(est, feats, rot, semanas, rotulo_cache):
    chave = hashlib.md5(json.dumps([est.nome, est.parametros, list(feats.columns), HORIZONTE, K_BARREIRA,
                                    [str(s) for s in semanas]], sort_keys=True, default=str).encode()).hexdigest()[:10]
    arq = CACHE_ML / f"{rotulo_cache}_{est.nome.split()[0]}_{chave}.pkl"
    if arq.exists():
        return pd.read_pickle(arq)
    print(f"  treinando {est.nome} em {len(semanas)} semanas (walk-forward)...")
    prev = prever_walkforward(est, feats, rot, semanas)
    CACHE_ML.mkdir(parents=True, exist_ok=True)
    prev.to_pickle(arq)
    return prev


def resumo(ret, ops, bench, n_tentativas):
    linhas = {}
    for nome in ret.columns:
        r = ret[nome]
        o = ops.get(nome, pd.DataFrame())
        curva = (1 + r).cumprod()
        linha = {
            "Semanas": len(r),
            "Ganhou do Bench": (r > bench).mean() if nome != NOME_BENCH else np.nan,
            "Média/semana": r.mean(),
            "Excesso médio": (r - bench).mean() if nome != NOME_BENCH else np.nan,
            "Retorno encadeado": curva.iloc[-1] - 1,
            "Queda máxima": queda_maxima(curva),
            "Sharpe (anual)": sharpe(r) * np.sqrt(52),
            "DSR": dsr(r, n_tentativas) if nome != NOME_BENCH else np.nan,
        }
        if nome != NOME_BENCH and len(o):
            linha["Operações/semana"] = len(o) / len(r)
            linha["% compras"] = (o["lado"] == 1).mean()
            linha["Acerto"] = (o["retorno"] > 0).mean()
            linha["Tempo posicionado"] = o["horas"].sum() / (168 * len(r))
        linhas[nome] = linha
    return pd.DataFrame(linhas).T


def fmt(t):
    t = t.copy()
    for c in t.columns:
        v = pd.to_numeric(t[c], errors="coerce")
        if v.notna().sum() != t[c].notna().sum():
            continue
        if c in ("Semanas",):
            t[c] = v.map(lambda x: f"{x:.0f}")
        elif c in ("Sharpe (anual)", "Operações/semana"):
            t[c] = v.map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        elif c in ("Ganhou do Bench", "DSR", "% compras", "Acerto", "Tempo posicionado", "Chance de ser sorte"):
            t[c] = v.map(lambda x: "-" if pd.isna(x) else f"{x:.0%}")
        else:
            t[c] = v.map(lambda x: "-" if pd.isna(x) else f"{x:+.2%}")
    return t.to_string()


def registrar_mlflow(nome_run, params, metricas, arquivos):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("empresa_ml_fase1")
        with mlflow.start_run(run_name=nome_run):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metricas.items() if pd.notna(v)})
            for a in arquivos:
                mlflow.log_artifact(str(a))
    except Exception as e:  # o MLflow nunca pode derrubar a simulação
        print(f"  (aviso: MLflow não registrou: {e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prova", action="store_true", help="roda também a prova das 20 semanas sorteadas")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    REL.mkdir(exist_ok=True)

    arq_params = REL / "ml_parametros.json"
    otim = json.loads(arq_params.read_text(encoding="utf-8")) if arq_params.exists() else {}
    params = otim.get("melhores", {})
    n_tentativas = max(1, otim.get("tentativas_total", 0)) + 3  # + os 3 modelos
    registro = REL / "registro_da_prova.csv"
    vezes_prova = len(pd.read_csv(registro)) if registro.exists() else 0
    n_tentativas += vezes_prova

    print("Carregando dados e montando features...")
    spot, funding, feats, rot = carregar()
    print(f"  {len(spot):,} candles de 1h | {feats.shape[1]} features | rótulos: "
          f"{(rot['rotulo'] == 1).mean():.0%} alta, {(rot['rotulo'] == -1).mean():.0%} baixa, "
          f"{(rot['rotulo'] == 0).mean():.0%} tempo")

    sem_estagio = semanas_do_periodo(*ESTAGIO)
    sem_treino = semanas_do_periodo(*TREINO)
    semanas_wf = sem_estagio + sem_treino
    sem_prova = list(pd.read_csv(REL / "semanas_sorteadas.csv", parse_dates=["semana"])["semana"])
    regras, regra = RegrasRH(), RegraParaTodos()
    bench_all = resultados_semanais(spot, [], "1h", regras, semanas_wf + sem_prova)[0][NOME_BENCH]

    turma = contratar_turma_ml(params)
    ret, ops, ret_prova, ops_prova, metas = {}, {}, {}, {}, {}
    for est in turma:
        prev = previsoes_com_cache(est, feats, rot, semanas_wf, "treino")
        r, o, dec, hist_meta = operar(est, prev, feats, rot, spot, funding, semanas_wf, regra, HORIZONTE)
        ret[est.nome], ops[est.nome], metas[est.nome] = r, o, hist_meta
        if args.prova:
            prev_p = previsoes_com_cache(est, feats, rot, sem_prova, "prova")
            rp, op, _, _ = operar(est, prev_p, feats, rot, spot, funding, sem_prova, regra, HORIZONTE,
                                  semana_de_estudo=False, historico_meta=hist_meta)
            ret_prova[est.nome], ops_prova[est.nome] = rp, op

    ret = pd.DataFrame(ret)
    ret[NOME_BENCH] = bench_all.reindex(ret.index)
    ret_estagio = ret.loc[ret.index < pd.Timestamp(TREINO[0])]
    ret_treino = ret.loc[ret.index >= pd.Timestamp(TREINO[0])]
    ops_treino = {n: o[o["semana"] >= pd.Timestamp(TREINO[0])] for n, o in ops.items()}

    print(f"\n--- ESTÁGIO 2020 ({len(ret_estagio)} semanas, não conta para o RH) ---")
    print(fmt(resumo(ret_estagio, {n: o[o['semana'] < pd.Timestamp(TREINO[0])] for n, o in ops.items()},
                     ret_estagio[NOME_BENCH], n_tentativas)))

    r_emp = rodar_empresa(ret_treino, regras)
    tab = resumo(ret_treino, ops_treino, ret_treino[NOME_BENCH], n_tentativas)
    tab["Status final"] = [r_emp["fichas"][n].status if n in r_emp["fichas"] else "-" for n in tab.index]
    print(f"\n--- TREINO 2021-2023 ({len(ret_treino)} semanas, cada uma testada separadamente) ---")
    print(fmt(tab.sort_values("Excesso médio", ascending=False)))
    emp = (1 + r_emp["empresa_semanal"]).cumprod()
    ben = (1 + ret_treino[NOME_BENCH]).cumprod()
    print(f"\nEmpresa: retorno {emp.iloc[-1] - 1:+.1%} | queda máxima {queda_maxima(emp):+.1%}")
    print(f"Benchmark: retorno {ben.iloc[-1] - 1:+.1%} | queda máxima {queda_maxima(ben):+.1%}")
    print("Diário do RH:")
    for linha in r_emp["diario_rh"] or ["Nenhuma mudança."]:
        print("  " + linha)
    print(f"\n(DSR leva em conta {n_tentativas} tentativas no total. Acima de 95% é evidência forte.)")

    base = REL / "ml_1h"
    tab.to_csv(f"{base}_treino_resumo.csv")
    ret.to_csv(f"{base}_semanas.csv")
    pd.concat(ops, names=["estagiario"]).to_csv(f"{base}_operacoes.csv")
    arquivos = [f"{base}_treino_resumo.csv", f"{base}_semanas.csv"]

    prova = None
    if args.prova:
        rp = pd.DataFrame(ret_prova)
        rp[NOME_BENCH] = bench_all.reindex(rp.index)
        prova = avaliar_prova(rp, r_emp["fichas"])
        extra = resumo(rp, ops_prova, rp[NOME_BENCH], n_tentativas + 1)
        prova = prova.join(extra[["Operações/semana", "% compras", "Acerto", "Tempo posicionado"]])
        print(f"\n--- PROVA: {len(rp)} semanas sorteadas de 2024-2025 (nada foi ajustado) ---")
        print(fmt(prova.sort_values("Excesso médio", ascending=False)))
        prova.to_csv(f"{base}_prova_resumo.csv")
        rp.to_csv(f"{base}_prova_semanas.csv")
        arquivos += [f"{base}_prova_resumo.csv"]
        nova = pd.DataFrame({"quando": [pd.Timestamp.now()], "parametros": [json.dumps(params)]})
        antigas = [pd.read_csv(registro)] if registro.exists() else []
        pd.concat(antigas + [nova]).to_csv(registro, index=False)

    for est in turma:
        m = tab.loc[est.nome].to_dict()
        metricas = {k: v for k, v in m.items() if isinstance(v, (int, float, np.floating))}
        if prova is not None:
            metricas["prova_vitorias"] = int(str(prova.loc[est.nome, "Vitórias"]).split("/")[0])
            metricas["prova_excesso_medio"] = prova.loc[est.nome, "Excesso médio"]
        registrar_mlflow(est.nome, {**est.parametros, "horizonte": HORIZONTE, "k_barreira": K_BARREIRA,
                                    **{f"regra_{k}": v for k, v in vars(regra).items()}},
                         {k.replace(" ", "_").replace("/", "_por_").replace("%", "pct"): v
                          for k, v in metricas.items()}, arquivos)
    print(f"\nTabelas em {REL} | experimentos no MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Lembrete: resultado passado não garante resultado futuro.\n")


if __name__ == "__main__":
    main()
