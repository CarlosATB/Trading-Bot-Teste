"""A empresa: cada semana é um teste. O gerente (RH) avalia os estagiários e redistribui o capital.

Como uma semana é testada (igual para todos, Benchmark incluído):
- A semana vai de segunda 00:00 a domingo 23:59 (UTC).
- Todo mundo começa a semana em dólar e termina em dólar.
- Os indicadores já chegam "aquecidos": são calculados com todo o histórico ANTERIOR,
  mas cada decisão usa só dados até o candle em que foi tomada (nunca o futuro).
- A posição decidida no fechamento de um candle vale a partir do candle seguinte.
- Cada troca de posição (entrar ou sair) custa taxa + slippage (~0,15%).
  O Benchmark compra no início da semana e vende no fim: paga 2 trocas.

Plano de carreira (decidido só com dados do passado):
- Período de experiência: nas primeiras semanas todos recebem o mesmo capital.
- A nota é o quanto o estagiário ganhou do Benchmark nas últimas N semanas.
- Promoção -> dobro de capital. Advertência -> metade. Demissão -> zero.
"""
from dataclasses import dataclass, field
from math import comb

import numpy as np
import pandas as pd

from dados import DURACAO
from estagiarios import Benchmark

PESOS = {"Promovido": 2.0, "Estagiário": 1.0, "Advertido": 0.5, "Demitido": 0.0}
NOME_BENCH = Benchmark.nome


@dataclass
class RegrasRH:
    taxa: float = 0.001            # 0,1% por troca (taxa da Binance)
    slippage: float = 0.0005       # 0,05% de piora no preço de execução
    semanas_experiencia: int = 26  # ~6 meses antes de qualquer promoção ou advertência
    janela_avaliacao: int = 26     # nota olha as últimas 26 semanas (~6 meses)
    avaliar_a_cada: int = 4        # o RH avalia a cada 4 semanas (~1 mês)
    margem_promocao: float = 0.05  # precisa ganhar do Benchmark por 5 p.p. na janela
    advertencias_para_demissao: int = 3
    queda_maxima_tolerada: float = 0.50  # demitido se cair 50% do topo

    @property
    def custo_troca(self):
        return self.taxa + self.slippage


@dataclass
class Ficha:
    nome: str
    status: str = "Estagiário"
    advertencias: int = 0
    historico: list = field(default_factory=list)


# ---------------------------------------------------------------- semanas

def semanas_do_periodo(inicio, fim):
    """Segundas-feiras de todas as semanas que cabem inteiras em [inicio, fim)."""
    seg = pd.date_range(pd.Timestamp(inicio), pd.Timestamp(fim), freq="W-MON")
    return [s for s in seg if s + pd.Timedelta("7D") <= pd.Timestamp(fim)]


def resultados_semanais(df, turma, intervalo, regras, semanas):
    """Roda cada estagiário (e o Benchmark) em cada semana, de forma independente.

    Devolve três tabelas (linhas = semanas, colunas = estagiários):
    retorno da semana já com custos, número de trocas e fração do tempo comprado.
    """
    esperado = pd.Timedelta("7D") / pd.Timedelta(DURACAO[intervalo])
    ret_candle = df["fechamento"].pct_change()
    # posição "segurada" em cada candle = decisão tomada no fechamento do candle anterior
    seguradas = {e.nome: e.posicoes(df).astype(float).shift(1) for e in turma}
    seguradas[NOME_BENCH] = None  # o Benchmark fica comprado a semana inteira

    ret, trocas, tempo = {}, {}, {}
    chave_semana = df.index.to_period("W-SUN").start_time
    grupos = {k: idx for k, idx in df.groupby(chave_semana).indices.items()}
    for s in semanas:
        idx = grupos.get(s)
        if idx is None or len(idx) < 0.95 * esperado or idx[0] == 0:
            continue  # semana sem dados suficientes (ex.: Binance fora do ar)
        r = ret_candle.iloc[idx].to_numpy()
        linha_r, linha_t, linha_c = {}, {}, {}
        for nome, pos in seguradas.items():
            h = np.ones(len(idx)) if pos is None else pos.iloc[idx].fillna(0.0).to_numpy()
            n_trocas = h[0] + np.abs(np.diff(h)).sum() + h[-1]  # entra, troca, sai no fim
            bruto = np.prod(1 + h * r)
            linha_r[nome] = bruto * (1 - regras.custo_troca) ** n_trocas - 1
            linha_t[nome] = n_trocas
            linha_c[nome] = h.mean()
        ret[s], trocas[s], tempo[s] = linha_r, linha_t, linha_c

    para_df = lambda d: pd.DataFrame.from_dict(d, orient="index").rename_axis("semana")
    return para_df(ret), para_df(trocas), para_df(tempo)


# ---------------------------------------------------------------- métricas

def queda_maxima(curva):
    return float((curva / curva.cummax() - 1).min())


def metricas_semanais(ret, trocas, tempo):
    """Resumo de cada estagiário a partir das semanas testadas."""
    bench = ret[NOME_BENCH]
    linhas = {}
    for nome in ret.columns:
        r = ret[nome]
        curva = (1 + r).cumprod()
        linhas[nome] = {
            "Semanas": len(r),
            "Ganhou do Bench": np.nan if nome == NOME_BENCH else (r > bench).mean(),
            "Média/semana": r.mean(),
            "Excesso médio": np.nan if nome == NOME_BENCH else (r - bench).mean(),
            "Retorno encadeado": curva.iloc[-1] - 1,
            "Queda máxima": queda_maxima(curva),
            "Sharpe (anual)": r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else np.nan,
            "Trocas/semana": trocas[nome].mean(),
            "Tempo comprado": tempo[nome].mean(),
        }
    return pd.DataFrame(linhas).T


# ---------------------------------------------------------------- RH (treino)

def rodar_empresa(ret, regras=RegrasRH()):
    """Treino: as semanas acontecem em sequência e o RH avalia periodicamente."""
    nomes = [c for c in ret.columns if c != NOME_BENCH]
    bench = ret[NOME_BENCH]
    fichas = {n: Ficha(n) for n in nomes}
    retorno_empresa, diario_rh = [], []

    for i, semana in enumerate(ret.index):
        # 1) Distribui o capital da semana conforme o status atual
        pesos = pd.Series({n: PESOS[f.status] for n, f in fichas.items()})
        if pesos.sum() == 0:
            retorno_empresa.append(0.0)  # todo mundo demitido: dinheiro parado
        else:
            retorno_empresa.append(float((ret.loc[semana, nomes] * pesos / pesos.sum()).sum()))

        # 2) Avaliação do RH (só com as semanas até esta)
        feitas = i + 1
        if feitas < regras.semanas_experiencia or feitas % regras.avaliar_a_cada:
            continue
        janela = slice(max(0, feitas - regras.janela_avaliacao), feitas)
        bench_janela = (1 + bench.iloc[janela]).prod() - 1
        for nome, ficha in fichas.items():
            if ficha.status == "Demitido":
                continue
            nota = (1 + ret[nome].iloc[janela]).prod() - 1 - bench_janela
            curva = (1 + ret[nome].iloc[:feitas]).cumprod()
            antes = ficha.status

            if queda_maxima(curva) < -regras.queda_maxima_tolerada:
                ficha.status = "Demitido"
                motivo = f"queda máxima de {queda_maxima(curva):.0%}"
            elif nota > regras.margem_promocao:
                ficha.status, ficha.advertencias = "Promovido", 0
                motivo = f"ganhou do Benchmark por {nota:+.1%} em {regras.janela_avaliacao} semanas"
            elif nota < 0:
                ficha.advertencias += 1
                if ficha.advertencias >= regras.advertencias_para_demissao:
                    ficha.status = "Demitido"
                    motivo = f"{ficha.advertencias} advertências seguidas"
                else:
                    ficha.status = "Advertido"
                    motivo = f"perdeu do Benchmark por {nota:+.1%}"
            else:
                ficha.status, ficha.advertencias = "Estagiário", 0
                motivo = "desempenho neutro"

            ficha.historico.append((f"{semana:%Y-%m-%d}", ficha.status, round(nota, 4)))
            if ficha.status != antes:
                diario_rh.append(f"{semana:%Y-%m-%d}: {nome}: {antes} -> {ficha.status} ({motivo})")

    return {
        "empresa_semanal": pd.Series(retorno_empresa, index=ret.index),
        "fichas": fichas,
        "diario_rh": diario_rh,
    }


# ---------------------------------------------------------------- prova

def sortear_semanas(inicio, fim, n, semente):
    """Sorteia n semanas inteiras do período, em ordem cronológica."""
    todas = semanas_do_periodo(inicio, fim)
    rng = np.random.default_rng(semente)
    escolhidas = rng.choice(len(todas), size=n, replace=False)
    return sorted(todas[i] for i in escolhidas)


def chance_de_sorte(vitorias, n):
    """Chance de ganhar `vitorias` ou mais de `n` semanas no cara ou coroa (p = 50%)."""
    return sum(comb(n, k) for k in range(vitorias, n + 1)) / 2 ** n


def avaliar_prova(ret, fichas_treino):
    """Aprovado = sobreviveu ao treino E ganhou do Benchmark na maioria das semanas."""
    bench = ret[NOME_BENCH]
    n = len(ret)
    linhas = {}
    for nome in ret.columns:
        if nome == NOME_BENCH:
            continue
        vit = int((ret[nome] > bench).sum())
        status_treino = fichas_treino[nome].status
        linhas[nome] = {
            "Vitórias": f"{vit}/{n}",
            "Excesso médio": (ret[nome] - bench).mean(),
            "Chance de ser sorte": chance_de_sorte(vit, n),
            "Status no treino": status_treino,
            "Aprovado": "SIM" if (vit > n / 2 and status_treino != "Demitido") else "não",
        }
    return pd.DataFrame(linhas).T
