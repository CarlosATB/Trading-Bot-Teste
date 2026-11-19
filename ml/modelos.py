"""Os estagiários de Machine Learning.

Todos fazem a mesma coisa: recebem as features de um candle e devolvem três probabilidades,
na ordem [bater embaixo, dar o tempo, bater em cima]. A decisão é da RegraParaTodos.
"""
import numpy as np
import pandas as pd

CLASSES = np.array([-1, 0, 1])


def _gpu_disponivel():
    try:
        import xgboost as xgb
        xgb.XGBClassifier(device="cuda", n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])
        return True
    except Exception:
        return False


class EstagiarioML:
    nome = "genérico"
    descricao = ""
    parametros: dict = {}

    def treinar(self, X, y, peso):
        raise NotImplementedError

    def prever(self, X):
        raise NotImplementedError


class Xavier(EstagiarioML):
    nome = "Xavier (XGBoost)"
    descricao = "Gradient boosting com XGBoost: árvores que corrigem os erros umas das outras."
    padrao = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 50,
              "subsample": 0.7, "colsample_bytree": 0.6, "reg_lambda": 5.0, "gamma": 0.0}

    def __init__(self, **params):
        self.parametros = {**self.padrao, **params}
        self.gpu = _gpu_disponivel()

    def treinar(self, X, y, peso):
        import xgboost as xgb
        self.modelo = xgb.XGBClassifier(
            objective="multi:softprob", tree_method="hist", device="cuda" if self.gpu else "cpu",
            n_jobs=-1, random_state=7, verbosity=0, **self.parametros)
        self.modelo.fit(X, y + 1, sample_weight=peso)
        return self

    def prever(self, X):
        return self.modelo.predict_proba(X)


class Luna(EstagiarioML):
    """LightGBM + meta-labeling.

    O modelo primário dá as probabilidades. O secundário (meta) aprende com o histórico das
    operações que a própria Luna propôs nas semanas anteriores: dada a situação, aquela entrada
    deu lucro? Ele não muda a direção, só confirma (ou veta) e ajusta o tamanho.
    """
    nome = "Luna (LightGBM + meta-labeling)"
    descricao = "LightGBM decide a direção; um segundo modelo aprende com os erros dela e decide se vale entrar."
    padrao = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
              "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 5.0}
    minimo_meta = 300  # operações passadas necessárias antes do meta começar a vetar

    def __init__(self, **params):
        self.parametros = {**self.padrao, **params}
        self.meta = None

    def treinar(self, X, y, peso):
        import lightgbm as lgb
        self.modelo = lgb.LGBMClassifier(objective="multiclass", n_jobs=-1, random_state=7,
                                         verbose=-1, **self.parametros)
        self.modelo.fit(X, y + 1, sample_weight=peso)
        return self

    def prever(self, X):
        return self.modelo.predict_proba(X)

    def treinar_meta(self, X_meta, acertou):
        """X_meta: features + probabilidades + valor esperado das entradas passadas; acertou: 1 se deu lucro."""
        import lightgbm as lgb
        if len(acertou) < self.minimo_meta or acertou.nunique() < 2:
            self.meta = None
            return
        self.meta = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                       min_child_samples=50, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.6, verbose=-1, random_state=7)
        self.meta.fit(X_meta, acertou)

    def confianca_meta(self, X_meta):
        if self.meta is None:
            return np.ones(len(X_meta))  # ainda sem histórico: não veta nada
        return self.meta.predict_proba(X_meta)[:, 1]


class Regua(EstagiarioML):
    """Não é estagiário: é a régua de ML. Se um modelo complexo não ganha dela, está decorando o passado."""
    nome = "Régua (regressão logística)"
    descricao = "Modelo linear simples. Serve de base para os outros."

    def __init__(self, C=0.05):
        self.parametros = {"C": C}

    def treinar(self, X, y, peso):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.modelo = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                    LogisticRegression(C=self.parametros["C"], max_iter=500))
        Xc = np.clip(X, -1e6, 1e6)
        self.modelo.fit(Xc, y + 1, logisticregression__sample_weight=peso)
        return self

    def prever(self, X):
        return self.modelo.predict_proba(np.clip(X, -1e6, 1e6))


def contratar_turma_ml(params=None):
    params = params or {}
    return [Xavier(**params.get("Xavier", {})), Luna(**params.get("Luna", {})), Regua()]
