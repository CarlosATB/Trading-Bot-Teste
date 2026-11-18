"""Relatório da empresa, semana a semana.

Uso:
    python main.py                          # BTC/USDT, 1h e 15min (dados reais da Binance)
    python main.py --simbolo ETHUSDT
    python main.py --intervalos 1h          # só 1h
    python main.py --sintetico              # dados falsos, sem internet

Etapas:
    Treino (2021 a 2023): todas as semanas em sequência, com o RH avaliando.
    Prova  (2024 e 2025): 20 semanas sorteadas UMA VEZ e guardadas em
           relatorios/semanas_sorteadas.csv. Rodadas seguintes reusam o mesmo sorteio,
           para ninguém "ajustar" a estratégia depois de ver as semanas da prova.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from dados import baixar_binance, dados_sinteticos, relatorio_buracos
from empresa import (NOME_BENCH, RegrasRH, avaliar_prova, metricas_semanais,
                     queda_maxima, resultados_semanais, rodar_empresa,
                     semanas_do_periodo, sortear_semanas)
from estagiarios import contratar_turma

PASTA_RELATORIOS = Path(__file__).parent / "relatorios"
TREINO = ("2021-01-04", "2024-01-01")   # começa na 1ª segunda-feira de 2021
PROVA = ("2024-01-01", "2025-12-29")    # semanas inteiras de 2024 e 2025
DADOS_DESDE = "2020-09-01"              # 4 meses antes do treino, para aquecer indicadores
DADOS_ATE = "2026-01-01"


def pct(x):
    return "-" if pd.isna(x) else f"{x:+.2%}"


def formatar(tabela):
    fmt = tabela.copy()
    for col in fmt.columns:
        convertido = pd.to_numeric(fmt[col], errors="coerce")
        if convertido.notna().sum() == fmt[col].notna().sum():
            fmt[col] = convertido.astype(float)
        if col in ("Semanas",):
            fmt[col] = fmt[col].map(lambda x: f"{x:.0f}")
        elif col in ("Sharpe (anual)", "Trocas/semana"):
            fmt[col] = fmt[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        elif col in ("Ganhou do Bench", "Tempo comprado", "Chance de ser sorte"):
            fmt[col] = fmt[col].map(lambda x: "-" if pd.isna(x) else f"{x:.0%}")
        elif fmt[col].dtype.kind == "f":
            fmt[col] = fmt[col].map(pct)
    return fmt.to_string()


def semanas_da_prova(n, semente, refazer):
    arquivo = PASTA_RELATORIOS / "semanas_sorteadas.csv"
    if arquivo.exists() and not refazer:
        return list(pd.read_csv(arquivo, parse_dates=["semana"])["semana"])
    semanas = sortear_semanas(*PROVA, n, semente)
    pd.DataFrame({"semana": semanas}).to_csv(arquivo, index=False)
    return semanas


def rodar(simbolo, intervalo, sintetico, semanas_prova, regras, saida):
    if sintetico:
        df = dados_sinteticos(intervalo, DADOS_DESDE, DADOS_ATE)
        origem = "DADOS SINTÉTICOS"
    else:
        df = baixar_binance(simbolo, intervalo, DADOS_DESDE, DADOS_ATE)
        origem = simbolo
    faltando, _ = relatorio_buracos(df, intervalo)

    titulo = f"{origem} | candles de {intervalo}"
    print(f"\n{'=' * 90}\n=== {titulo}  ({len(df):,} candles, {faltando} faltando na Binance)\n{'=' * 90}")
    turma = contratar_turma()

    # ---- Treino
    ret, trocas, tempo = resultados_semanais(df, turma, intervalo, regras, semanas_do_periodo(*TREINO))
    r = rodar_empresa(ret, regras)
    tabela = metricas_semanais(ret, trocas, tempo).sort_values("Excesso médio", ascending=False)
    tabela["Status final"] = [r["fichas"][n].status if n in r["fichas"] else "-" for n in tabela.index]
    print(f"\n--- TREINO 2021-2023: {len(ret)} semanas, cada uma testada separadamente ---")
    print(formatar(tabela))

    emp = (1 + r["empresa_semanal"]).cumprod()
    ben = (1 + ret[NOME_BENCH]).cumprod()
    print("\nEmpresa (RH redistribuindo capital) vs Benchmark, semanas encadeadas:")
    print(f"  Empresa:   retorno {pct(emp.iloc[-1] - 1)} | queda máxima {pct(queda_maxima(emp))}")
    print(f"  Benchmark: retorno {pct(ben.iloc[-1] - 1)} | queda máxima {pct(queda_maxima(ben))}")
    print("\nDiário do RH:")
    for linha in r["diario_rh"] or ["Nenhuma mudança."]:
        print("  " + linha)

    # ---- Prova
    ret_p, trocas_p, tempo_p = resultados_semanais(df, turma, intervalo, regras, semanas_prova)
    prova = avaliar_prova(ret_p, r["fichas"])
    prova = prova.join(metricas_semanais(ret_p, trocas_p, tempo_p)[["Trocas/semana", "Tempo comprado"]])
    print(f"\n--- PROVA: {len(ret_p)} semanas sorteadas de 2024-2025 (nada foi ajustado) ---")
    print(formatar(prova.sort_values("Excesso médio", ascending=False)))

    base = saida / f"{'SINTETICO' if sintetico else origem}_{intervalo}"
    tabela.to_csv(f"{base}_treino_resumo.csv")
    ret.to_csv(f"{base}_treino_semanas.csv")
    prova.to_csv(f"{base}_prova_resumo.csv")
    ret_p.to_csv(f"{base}_prova_semanas.csv")
    pd.Series(r["diario_rh"], name="evento").to_csv(f"{base}_diario_rh.csv", index=False)
    return prova


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simbolo", default="BTCUSDT")
    p.add_argument("--intervalos", nargs="+", default=["1h", "15m"], choices=["15m", "1h", "4h"])
    p.add_argument("--semanas-prova", type=int, default=20)
    p.add_argument("--semente", type=int, default=2024)
    p.add_argument("--refazer-sorteio", action="store_true",
                   help="sorteia semanas novas (só use antes de olhar qualquer resultado da prova)")
    p.add_argument("--sintetico", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # acentos e σ no terminal do Windows

    PASTA_RELATORIOS.mkdir(exist_ok=True)
    semanas = semanas_da_prova(args.semanas_prova, args.semente, args.refazer_sorteio)
    print("Semanas da prova:", ", ".join(f"{s:%Y-%m-%d}" for s in semanas))

    for intervalo in args.intervalos:
        rodar(args.simbolo, intervalo, args.sintetico, semanas, RegrasRH(), PASTA_RELATORIOS)

    print(f"\nTabelas salvas em {PASTA_RELATORIOS}")
    print("Lembrete: resultado passado não garante resultado futuro. "
          "Use isto para estudar, não como sinal para investir.\n")


if __name__ == "__main__":
    main()
