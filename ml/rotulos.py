"""Rótulos triple-barrier (López de Prado, 2018).

Para cada candle t, imaginamos uma entrada no fechamento de t e olhamos as próximas H horas:
    +1 -> o preço bateu primeiro a barreira de CIMA (alta de U%)
    -1 -> o preço bateu primeiro a barreira de BAIXO (queda de L%)
     0 -> nenhuma das duas em H horas (barreira de TEMPO)
As barreiras se ajustam à volatilidade do momento: largura = k * vol_horária * raiz(H).
Também guardamos quanto tempo levou até o toque e o retorno na saída.
Os rótulos olham o futuro DE PROPÓSITO (é o que o modelo aprende a prever); por isso o
walk-forward descarta (purga) do treino todo rótulo que termine depois do início da semana operada.
"""
import numpy as np
import pandas as pd

from ml.features import volatilidade_horaria


def triple_barrier(spot, horizonte=24, k=1.0, vol_min=0.002):
    c = spot["fechamento"].to_numpy()
    hi = spot["maxima"].to_numpy()
    lo = spot["minima"].to_numpy()
    ab = spot["abertura"].to_numpy()
    n = len(c)
    sigma = np.maximum(volatilidade_horaria(spot).to_numpy(), vol_min)
    largura = k * sigma * np.sqrt(horizonte)          # em log
    sup = c * np.exp(largura)
    inf = c * np.exp(-largura)

    rotulo = np.zeros(n, dtype=np.int8)
    tempo = np.full(n, horizonte, dtype=np.int16)
    resolvido = np.zeros(n, dtype=bool)
    for j in range(1, horizonte + 1):
        idx = np.arange(n - j)
        h, l = hi[j:], lo[j:]
        toca_sup = h >= sup[idx]
        toca_inf = l <= inf[idx]
        ambos = toca_sup & toca_inf
        # se as duas barreiras caem no mesmo candle, decide pela direção do candle
        sobe = ab[j:] <= c[j:]
        novo_sup = ~resolvido[idx] & toca_sup & (~ambos | sobe)
        novo_inf = ~resolvido[idx] & toca_inf & (~ambos | ~sobe)
        rotulo[idx[novo_sup]] = 1
        rotulo[idx[novo_inf]] = -1
        tempo[idx[novo_sup | novo_inf]] = j
        resolvido[idx[novo_sup | novo_inf]] = True

    ret_saida = np.full(n, np.nan)
    fim = np.arange(n) + horizonte
    ok = fim < n
    ret_saida[ok] = np.log(c[fim[ok]] / c[ok])         # retorno se deu tempo
    ret_saida[rotulo == 1] = largura[rotulo == 1]
    ret_saida[rotulo == -1] = -largura[rotulo == -1]
    valido = ok | resolvido
    valido[n - horizonte:] &= resolvido[n - horizonte:]

    return pd.DataFrame({
        "rotulo": rotulo, "tempo": tempo, "ret_saida": ret_saida, "largura": largura,
        "fim_rotulo": spot.index + pd.to_timedelta(tempo.astype(int) + 1, unit="h"),
        "valido": valido,
    }, index=spot.index)
