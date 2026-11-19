"""Optuna ANINHADO (v0.2): os hiperparâmetros usados em cada ano são escolhidos só com anos anteriores.

Para operar o ano Y, o Optuna testa configurações treinando com dados até o início de cada
semestre de Y-1 e simulando as operações nesse semestre (regra para todos + custos).
Nada do ano Y, nem da prova (2024-2025), entra na escolha.

  ano 2020 (estágio) -> escolhido com 2019
  ano 2021           -> escolhido com 2020
  ano 2022           -> escolhido com 2021
  ano 2023           -> escolhido com 2022
  prova 2024-2025    -> escolhido com 2023

Objetivo (v0.2): retorno ajustado ao risco, o Sharpe das semanas simuladas, e não a precisão
das probabilidades. Ficar parado dá Sharpe zero, então proteção sozinha não é premiada.

Uso: python otimizar.py --tentativas 20
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd

from empresa import semanas_do_periodo
from main_ml import HORIZONTE, REL, RAIZ, carregar
from ml.modelos import Luna, Xavier
from ml.regimes import capture_ratio
from ml.regra import RegraParaTodos, decidir
from ml.simulador import simular_semana
from ml.walkforward import contexto_da_regra, pesos_por_regime, regime_semanal_por_candle, unicidade

ANOS = {"2020": 2019, "2021": 2020, "2022": 2021, "2023": 2022, "prova": 2023}


def espaco(trial, nome):
    if nome == "Xavier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 5, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tentativas", type=int, default=20)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    spot, funding, feats, rot = carregar()
    regra = RegraParaTodos()
    custo_lado = regra.taxa + regra.spread_slippage
    X = feats.to_numpy()
    y = rot["rotulo"].to_numpy()
    u = unicidade(rot).to_numpy()
    fim = rot["fim_rotulo"].to_numpy()
    valido = rot["valido"].to_numpy()
    reg = regime_semanal_por_candle(spot).astype(str).to_numpy()
    bench_sem = {}

    def simular_bloco(classe, params, ini, fim_b):
        """Treina até `ini` e opera as semanas de [ini, fim_b) com a regra para todos."""
        tr = valido & (fim <= np.datetime64(ini))
        idade = (ini - rot.index[tr]).days.to_numpy()
        w = u[tr] * 0.5 ** (idade / 365) * pesos_por_regime(reg[tr])
        modelo = classe(**params).treinar(X[tr], y[tr], w / w.mean())
        rets, bench = [], []
        for s in semanas_do_periodo(ini, fim_b):
            idx = (rot.index >= s) & (rot.index < s + pd.Timedelta("7D"))
            if idx.sum() < 160:
                continue
            probs = pd.DataFrame(modelo.prever(X[idx]), index=rot.index[idx],
                                 columns=["p_baixa", "p_tempo", "p_alta"])
            ctx = contexto_da_regra(rot, feats, spot, s, pd.Series(tr, index=rot.index))
            dec = decidir(probs, ctx, regra)
            dec["largura"] = ctx["largura"]
            precos = spot.loc[idx, ["abertura", "maxima", "minima", "fechamento"]]
            r, _ = simular_semana(precos, dec, funding, custo_lado, HORIZONTE)
            rets.append(r)
            if s not in bench_sem:
                c = spot["fechamento"]
                bench_sem[s] = c[idx].iloc[-1] / c[rot.index < s].iloc[-1] * (1 - custo_lado) ** 2 - 1
            bench.append(bench_sem[s])
        return rets, bench

    def avaliar(classe, params, ano_validacao):
        rets, bench = [], []
        for ini, fim_b in ((f"{ano_validacao}-01-01", f"{ano_validacao}-07-01"),
                           (f"{ano_validacao}-07-01", f"{ano_validacao + 1}-01-01")):
            r, b = simular_bloco(classe, params, pd.Timestamp(ini), pd.Timestamp(fim_b))
            rets += r
            bench += b
        r = np.array(rets)
        sharpe = r.mean() / r.std(ddof=1) * np.sqrt(52) if r.std(ddof=1) > 0 else 0.0
        return float(sharpe), float(capture_ratio(pd.Series(r), pd.Series(bench)))

    saida = {"por_ano": {}, "tentativas": {}, "detalhes": {}}
    t0 = time.time()
    for nome, classe in (("Xavier", Xavier), ("Luna", Luna)):
        saida["por_ano"][nome], saida["detalhes"][nome] = {}, {}
        for ano, ano_val in ANOS.items():
            base, cap_base = avaliar(classe, {}, ano_val)
            estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))

            def objetivo(trial):
                sh, cap = avaliar(classe, espaco(trial, nome), ano_val)
                trial.set_user_attr("capture", cap)
                return sh

            estudo.optimize(objetivo, n_trials=args.tentativas, show_progress_bar=False)
            usar = estudo.best_params if estudo.best_value > base else {}
            saida["por_ano"][nome][ano] = usar
            saida["tentativas"][f"{nome}_{ano}"] = len(estudo.trials)
            saida["detalhes"][nome][ano] = {
                "validado_em": ano_val, "sharpe_padrao": base, "sharpe_melhor": estudo.best_value,
                "capture_melhor": estudo.best_trial.user_attrs.get("capture"), "capture_padrao": cap_base}
            print(f"{nome} {ano} (validado em {ano_val}): Sharpe padrão {base:+.2f} | melhor {estudo.best_value:+.2f} "
                  f"-> {'parâmetros novos' if usar else 'padrão'} ({time.time() - t0:.0f}s)", flush=True)
            try:
                import mlflow
                mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
                mlflow.set_experiment("empresa_ml_otimizacao_v02")
                with mlflow.start_run(run_name=f"optuna_{nome}_{ano}"):
                    mlflow.log_params({**estudo.best_params, "ano": ano, "validado_em": ano_val})
                    mlflow.log_metrics({"sharpe_melhor": estudo.best_value, "sharpe_padrao": base,
                                        "tentativas": len(estudo.trials)})
            except Exception as e:
                print(f"  (aviso: MLflow não registrou: {e})")

    saida["tentativas_total"] = sum(saida["tentativas"].values())
    REL.mkdir(exist_ok=True)
    (REL / "ml_parametros.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Salvo em {REL / 'ml_parametros.json'}")


if __name__ == "__main__":
    main()
