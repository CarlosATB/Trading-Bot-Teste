"""Uma regra para todos (definida pelo dono).

Nenhum integrante opera simplesmente porque identificou uma tendência. Antes de cada entrada,
todos precisam considerar:
  * Probabilidade de atingir a barreira de lucro.
  * Probabilidade de atingir a barreira de prejuízo.
  * Tempo até atingir uma das barreiras.
  * Taxas, spread e slippage.
  * Relação entre retorno esperado e risco.

O modelo de cada estagiário só fornece as três probabilidades. A decisão (comprar, vender ou
ficar fora) e o tamanho da posição saem SEMPRE desta regra, igual para todos.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegraParaTodos:
    taxa: float = 0.001             # 0,1% por lado
    spread_slippage: float = 0.0005  # 0,05% por lado
    rr_minimo: float = 0.10          # retorno esperado precisa ser >= 10% do risco assumido
    prob_minima: float = 0.40        # a barreira a favor precisa ter pelo menos 40% de chance
    kelly_fracao: float = 0.5        # meio Kelly
    vol_alvo_anual: float = 0.60     # reduz a posição quando o mercado está mais agitado que isso
    horas_por_funding: int = 8

    @property
    def custo_ida_volta(self):
        return 2 * (self.taxa + self.spread_slippage)


def decidir(probs, contexto, regra=RegraParaTodos()):
    """Aplica a regra a cada candle.

    probs: DataFrame com colunas p_baixa, p_tempo, p_alta (somam 1).
    contexto: DataFrame com
        largura       -> largura das barreiras (log), igual para cima e para baixo
        ret_tempo     -> retorno médio quando dá o tempo (aprendido no treino)
        t_alta, t_baixa, t_tempo -> horas médias até cada desfecho (aprendidas no treino)
        funding       -> funding atual do perpétuo (por 8h)
        vol_anual     -> volatilidade anualizada atual
        horas_restantes -> horas até o fim da semana
    Devolve: lado (+1 compra, -1 vende, 0 fora), tamanho (0 a 1), valor esperado e motivo.
    """
    pa, pb, pt = probs["p_alta"], probs["p_baixa"], probs["p_tempo"]
    U = np.exp(contexto["largura"]) - 1          # ganho se bater em cima
    L = 1 - np.exp(-contexto["largura"])          # perda se bater embaixo
    m = contexto["ret_tempo"]
    custo = regra.custo_ida_volta

    # 3) tempo esperado até uma das barreiras
    tempo_esp = pa * contexto["t_alta"] + pb * contexto["t_baixa"] + pt * contexto["t_tempo"]
    n_fundings = tempo_esp / regra.horas_por_funding

    # 1, 2 e 4) valor esperado líquido de custos para cada lado
    ev_compra = pa * U - pb * L + pt * m - custo
    # vendido no perpétuo: funding positivo = comprados pagam vendidos
    ev_venda = pb * L - pa * U - pt * m - custo + contexto["funding"] * n_fundings

    # 5) relação retorno esperado / risco (risco = perder na barreira + custos)
    rr_compra = ev_compra / (L + custo)
    rr_venda = ev_venda / (U + custo)

    ok_compra = (ev_compra > 0) & (rr_compra >= regra.rr_minimo) & (pa >= regra.prob_minima)
    ok_venda = (ev_venda > 0) & (rr_venda >= regra.rr_minimo) & (pb >= regra.prob_minima)
    cabe_na_semana = tempo_esp <= contexto["horas_restantes"]

    lado = pd.Series(0, index=probs.index, dtype=int)
    lado[ok_compra & cabe_na_semana & (ev_compra >= ev_venda)] = 1
    lado[ok_venda & cabe_na_semana & (ev_venda > ev_compra)] = -1

    # tamanho: meio Kelly de uma aposta com ganho U e perda L, limitado pela volatilidade
    p_ganho = np.where(lado == 1, pa, pb)
    p_perda = np.where(lado == 1, pb, pa)
    ganho = np.where(lado == 1, U, L)
    perda = np.where(lado == 1, L, U)
    kelly = (p_ganho * ganho - p_perda * perda) / (ganho * perda)
    limite_vol = np.minimum(1.0, regra.vol_alvo_anual / contexto["vol_anual"].clip(lower=1e-6))
    tamanho = np.clip(regra.kelly_fracao * kelly, 0, 1) * limite_vol
    tamanho = pd.Series(np.where(lado != 0, tamanho, 0.0), index=probs.index)

    ev = pd.Series(np.where(lado == 1, ev_compra, np.where(lado == -1, ev_venda, 0.0)), index=probs.index)
    return pd.DataFrame({"lado": lado, "tamanho": tamanho, "ev": ev, "tempo_esperado": tempo_esp,
                         "ev_compra": ev_compra, "ev_venda": ev_venda})
