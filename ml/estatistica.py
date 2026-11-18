"""Estatística para separar habilidade de sorte (Bailey e López de Prado).

PSR (Probabilistic Sharpe Ratio): chance de o Sharpe verdadeiro ser maior que um valor de referência,
levando em conta o tamanho da amostra, a assimetria e as caudas gordas dos retornos.
DSR (Deflated Sharpe Ratio): o PSR usando como referência o melhor Sharpe que se esperaria obter
POR SORTE depois de testar N configurações (cada tentativa do Optuna conta).
"""
import numpy as np
from scipy import stats


def sharpe(r):
    r = np.asarray(r, dtype=float)
    return r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0


def psr(r, sr_ref=0.0):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 3:
        return np.nan
    sr = sharpe(r)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_ref) * np.sqrt(n - 1) / den))


def sharpe_esperado_por_sorte(n_tentativas, n_obs):
    """Maior Sharpe esperado entre N tentativas sem habilidade nenhuma (Sharpe verdadeiro = 0)."""
    if n_tentativas <= 1:
        return 0.0
    gama = 0.5772156649
    z = stats.norm.ppf
    desvio = np.sqrt(1 / (n_obs - 1))  # desvio do Sharpe estimado quando o verdadeiro é zero
    return desvio * ((1 - gama) * z(1 - 1 / n_tentativas) + gama * z(1 - 1 / (n_tentativas * np.e)))


def dsr(r, n_tentativas):
    return psr(r, sharpe_esperado_por_sorte(n_tentativas, len(r)))
