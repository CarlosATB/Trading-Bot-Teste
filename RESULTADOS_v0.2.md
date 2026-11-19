# Segundo resultado (v0.2)

**Resumo:** nenhum bot foi aprovado ainda, mas a **Luna** deu o maior passo até aqui. Ela foi o primeiro bot a terminar a prova com diferença média positiva contra o BTC (+0,05% por semana) e passou a capturar 29% das altas, contra 6% na v0.1. O **Xavier** piorou: o Optuna achou parâmetros que pareciam ótimos no ano anterior e não funcionaram no ano seguinte (overfitting). A avaliação por regime confirmou o diagnóstico do Carlos: na v0.1 os bots quase não acompanhavam as altas.

## O que o Carlos mudou

| # | Mudança | Como foi implementada |
|---|---|---|
| 1 | Avaliação separada por regime de mercado | Cada semana é classificada pelo retorno do BTC: alta forte (≥ +5%), alta moderada (+1,5% a +5%), lateral, queda moderada (-5% a -1,5%) e queda forte (≤ -5%). A classificação serve só para avaliar; nenhum bot vê o regime da semana em que opera. |
| 2 | Capture Ratio e Downside Capture | Retorno médio do bot nas semanas de alta (ou de queda) dividido pelo retorno médio do BTC nessas semanas. |
| 3 | Xavier: otimizar retorno ajustado ao risco | O Optuna deixou de medir a precisão das probabilidades e passou a maximizar o Sharpe das semanas simuladas, com custos. Ficar parado dá Sharpe zero, então proteger sozinho não é premiado. |
| 4 | Luna: mais sensibilidade às altas | 9 features novas de tendência (eficiência de Kaufman, inclinação de 168h, retorno de 6 semanas, distância do topo e do fundo, entre outras). O veto do meta-labeling passou a ser escolhido ano a ano com o resultado do ano anterior e não encolhe mais a posição. |
| 5 | Treino com altas fortes | Histórico desde ago/2017, com as altas de 2017 e 2019. Os exemplos são pesados para que cada regime tenha o mesmo peso total, sem favorecer compra nem venda. |
| 6 | Optuna com walk-forward aninhado | Os parâmetros de cada ano são escolhidos só com o ano anterior (2021 com 2020, 2022 com 2021, e assim por diante). A prova usa parâmetros escolhidos com 2023. |

A regra para todos, os custos (0,15% por lado), as regras do RH e as 20 semanas sorteadas da prova não mudaram.

## Prova: 20 semanas sorteadas (2024 e 2025)

| Bot | Semanas ganhando do BTC | Diferença média por semana | Capture Ratio (altas) | Downside Capture (quedas) | Aprovado |
|---|:-:|:-:|:-:|:-:|:-:|
| **Luna** (LightGBM + meta) | 13/20 (v0.1: 13) | **+0,05%** (v0.1: -0,08%) | **+29%** (v0.1: +6%) | +15% (v0.1: -7%) | não |
| **Xavier** (XGBoost) | 13/20 (v0.1: 14) | -0,64% (v0.1: -0,60%) | -5% (v0.1: -3%) | +2% (v0.1: +4%) | não |
| Régua (logística) | 11/20 (v0.1: 10) | -1,72% (v0.1: -1,81%) | -45% (v0.1: -26%) | 0% (v0.1: +26%) | não |

*Capture: 100% = acompanhou a alta inteira do BTC. Downside: 100% = caiu junto; 0% = evitou a queda; negativo = lucrou com ela.*

### Prova por regime (retorno médio por semana)

| Regime (BTC na média) | Semanas | Luna | Xavier | Régua |
|---|:-:|:-:|:-:|:-:|
| Alta forte (+12,3%) | 4 | **+1,80%** | +0,33% | -2,92% |
| Alta moderada (+3,4%) | 2 | **+2,37%** | +2,03% | -0,58% |
| Lateral (-0,1%) | 4 | **+0,85%** | -0,81% | -3,26% |
| Queda moderada (-3,4%) | 7 | -0,72% | -0,43% | -1,51% |
| Queda forte (-7,7%) | 3 | -0,17% | -1,04% | **+3,57%** |

*São só 2 a 7 semanas por regime: serve de pista, não de prova.*

## Treino: 2021 a 2023 (156 semanas)

| Bot | Retorno no período | Queda máxima | Capture (altas) | Downside (quedas) | Status no RH |
|---|:-:|:-:|:-:|:-:|:-:|
| Régua (logística) | **+48,3%** | -62,2% | -5% | **-17%** | demitida em fev/2023 |
| **Luna** | +20,4% (v0.1: -5,2%) | -37,9% | +3% | -2% | demitida em abr/2023 |
| **Xavier** | -59,3% (v0.1: +20,7%) | -64,0% | -3% | +13% | demitido em nov/2021 |
| *Benchmark (BTC)* | *-19,8%* | *-78,9%* | *100%* | *100%* | *a régua* |

A empresa, com o RH redistribuindo o capital, rendeu **+47,3%** no treino (v0.1: +33,9%), com queda máxima de 31,1%.

**Atenção na comparação do treino:** na v0.1, o Optuna escolheu os parâmetros olhando 2022 e 2023, então o treino da v0.1 era otimista. Na v0.2 o treino é honesto, porque cada ano usa só parâmetros escolhidos antes dele. Parte da queda do Xavier vem daí. A prova é a comparação justa entre as versões.

## O que aprendemos

- **A Luna melhorou nas altas.** Na prova ela ganhou do BTC em semanas de alta forte, alta moderada e lateral. Ainda perde um pouco nas quedas moderadas.
- **O Xavier sofreu overfitting do Optuna.** Ele chegou a Sharpe +1,4 na validação de cada ano e -1,0 no ano seguinte. Otimizar o Sharpe de só 52 semanas é barulhento demais.
- **A Régua é especialista em queda forte.** Ela ganha bem vendendo (short) nas quedas fortes e perde em todo o resto.
- **Capture Ratio ainda baixo.** Mesmo a Luna fica com menos de 1/3 das altas. Os bots ainda preferem ficar de fora.
- **Ainda não há prova de habilidade.** O Deflated Sharpe ficou abaixo de 3% para todos. Ele já desconta 267 tentativas feitas desde a v0.1.

## Sugestões para a v0.3

1. Xavier: voltar a otimizar a precisão das probabilidades (mais estável) ou validar em 2 anos em vez de 1.
2. Comitê: combinar a Luna nas altas e laterais com a Régua nas quedas fortes.
3. Fase 2: Lucas (LSTM) e Tereza (Transformer).

Detalhes completos: [`relatorios/saida_ml_v0.2.txt`](relatorios/saida_ml_v0.2.txt). Primeiro resultado: [RESULTADOS.md](RESULTADOS.md).

> Resultado passado não garante resultado futuro. Projeto de estudo, sem dinheiro real.
