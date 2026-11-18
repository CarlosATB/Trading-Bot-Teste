"""Optuna: escolhe os hiperparâmetros do Xavier e da Luna usando SÓ dados até 2023.

Validação cruzada "purgada" e em ordem no tempo: para cada bloco de validação (semestres de
2022 e 2023), o modelo treina só com rótulos que terminaram antes do bloco começar. Nada de 2024
ou 2025 entra aqui, então a prova continua intocada.

Uso: python otimizar.py --tentativas 30
O resultado vai para relatorios/ml_parametros.json (lido pelo main_ml.py) e para o MLflow.
O número total de tentativas entra no Deflated Sharpe: quanto mais se testa, mais exigente fica.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from main_ml import REL, RAIZ, carregar
from ml.modelos import Luna, Xavier
from ml.walkforward import unicidade

BLOCOS = [("2022-01-03", "2022-07-04"), ("2022-07-04", "2023-01-02"),
          ("2023-01-02", "2023-07-03"), ("2023-07-03", "2024-01-01")]


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
    p.add_argument("--tentativas", type=int, default=30)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _, _, feats, rot = carregar()
    X = feats.to_numpy()
    y = rot["rotulo"].to_numpy() + 1
    u = unicidade(rot).to_numpy()
    fim = rot["fim_rotulo"].to_numpy()
    valido = rot["valido"].to_numpy()

    def avaliar(classe, params):
        perdas = []
        for ini, fim_b in BLOCOS:
            ini, fim_b = pd.Timestamp(ini), pd.Timestamp(fim_b)
            tr = valido & (fim <= np.datetime64(ini))
            va = valido & (rot.index >= ini) & (fim <= np.datetime64(fim_b))
            idade = (ini - rot.index[tr]).days.to_numpy()
            w = u[tr] * 0.5 ** (idade / 365)
            m = classe(**params).treinar(X[tr], y[tr] - 1, w / w.mean())
            perdas.append(log_loss(y[va], m.prever(X[va]), sample_weight=u[va], labels=[0, 1, 2]))
        return float(np.mean(perdas))

    saida = {"melhores": {}, "tentativas": {}, "perda_padrao": {}}
    for nome, classe in (("Xavier", Xavier), ("Luna", Luna)):
        base = avaliar(classe, {})
        print(f"{nome}: perda com parâmetros padrão = {base:.4f}")
        estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
        estudo.optimize(lambda t: avaliar(classe, espaco(t, nome)), n_trials=args.tentativas,
                        show_progress_bar=False,
                        callbacks=[lambda s, t: print(f"  tentativa {t.number + 1}: {t.value:.4f} "
                                                      f"(melhor {s.best_value:.4f})")])
        melhor = estudo.best_params if estudo.best_value < base else {}
        print(f"{nome}: melhor perda {estudo.best_value:.4f} -> {'usa os parâmetros novos' if melhor else 'fica com o padrão'}")
        saida["melhores"][nome] = melhor
        saida["tentativas"][nome] = len(estudo.trials)
        saida["perda_padrao"][nome] = base
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
            mlflow.set_experiment("empresa_ml_otimizacao")
            with mlflow.start_run(run_name=f"optuna_{nome}"):
                mlflow.log_params({k: v for k, v in estudo.best_params.items()})
                mlflow.log_metrics({"melhor_logloss": estudo.best_value, "logloss_padrao": base,
                                    "tentativas": len(estudo.trials)})
        except Exception as e:
            print(f"  (aviso: MLflow não registrou: {e})")

    saida["tentativas_total"] = sum(saida["tentativas"].values())
    REL.mkdir(exist_ok=True)
    (REL / "ml_parametros.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Salvo em {REL / 'ml_parametros.json'}")


if __name__ == "__main__":
    main()
