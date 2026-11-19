# Primeiro resultado (v0.1)

**Resumo:** nenhum bot foi aprovado. Os bots de Machine Learning (Xavier e Luna) foram bem melhores que os de regras fixas e protegeram muito melhor nas quedas. Mesmo assim, ainda não ganharam do buy and hold na média, porque ficam de fora nas semanas de alta forte.

Teste com BTC/USDT em candles de 1h e custo de 0,15% por troca. Cada semana é um teste separado: o bot começa e termina em dólar. A prova usa 20 semanas sorteadas de 2024 e 2025, rodadas uma única vez e sem nenhum ajuste depois.

## Prova: 20 semanas sorteadas (2024 e 2025)

| Bot | Tipo | Semanas ganhando do Benchmark | Diferença média por semana | Chance de ser sorte | Aprovado |
|---|---|:-:|:-:|:-:|:-:|
| **Xavier** (XGBoost) | Machine Learning | **14 de 20** | -0,60% | 6% | não |
| **Luna** (LightGBM + meta-labeling) | Machine Learning | 13 de 20 | **-0,08%** | 13% | não |
| RSI 14 (30/70) | Regra fixa | 13 de 20 | -0,73% | 13% | não |
| Reversão 20/2σ | Regra fixa | 11 de 20 | -1,19% | 41% | não |
| Breakout 55/20 | Regra fixa | 10 de 20 | +0,38% | 59% | não |
| Médias 50/200 | Regra fixa | 10 de 20 | +0,25% | 59% | não |
| Régua (regressão logística) | ML simples, referência | 10 de 20 | -1,81% | 59% | não |
| Médias 20/50 | Regra fixa | 8 de 20 | -0,57% | 87% | não |

*"Chance de ser sorte" é a probabilidade de um bot que joga cara ou coroa ganhar esse número de semanas ou mais.*

## Treino: 2021 a 2023 (156 semanas)

| Bot | Retorno no período | Queda máxima | Status no RH |
|---|:-:|:-:|:-:|
| **Xavier** (XGBoost) | **+20,7%** | -41,0% | demitido em fev/2023 |
| Régua (regressão logística) | +14,9% | -55,1% | demitido em fev/2023 |
| **Luna** (LightGBM + meta-labeling) | -5,2% | **-18,5%** | demitida em mar/2023 |
| Médias 50/200 | -16,4% | -59,8% | demitido em dez/2021 |
| Médias 20/50 | -54,8% | -73,5% | demitido em set/2021 |
| Breakout 55/20 | -64,7% | -73,1% | demitido em set/2021 |
| RSI 14 (30/70) | -73,0% | -80,6% | demitido em nov/2021 |
| Reversão 20/2σ | -81,4% | -81,3% | demitido em nov/2021 |
| *Benchmark (buy and hold)* | *-19,8%* | *-78,9%* | *a régua* |

A empresa de ML, com o RH redistribuindo o capital entre os bots, rendeu **+33,9%** no treino, com queda máxima de 26,4%. O Benchmark perdeu 19,8%, com queda máxima de 78,9%. Esse número do treino é otimista, porque os parâmetros foram escolhidos olhando 2022 e 2023. A prova não tem esse viés.

## O que aprendemos

- **Regra fixa não funciona aqui.** Em candles de 15min foi ainda pior: os custos engoliram o lucro.
- **O ML aprendeu a se defender.** Em 2022, quando o BTC caiu 65%, ele ficou de fora ou vendeu.
- **O ponto fraco é perder as altas fortes.** Os bots ganham pouco em muitas semanas e perdem muito nas semanas de alta. Esse é o próximo alvo.
- **Ainda não há prova de habilidade.** O Deflated Sharpe, que desconta as 63 tentativas feitas, ficou abaixo de 5% para todos.

## Próximos passos

1. Atacar o ponto fraco nas semanas de alta.
2. Fase 2: Lucas (LSTM) e Tereza (Transformer).
3. Fase 3: Paulo (Reinforcement Learning).

Detalhes completos: [`relatorios/saida_ml_fase1.txt`](relatorios/saida_ml_fase1.txt) (ML) e [`relatorios/saida_BTCUSDT.txt`](relatorios/saida_BTCUSDT.txt) (regras fixas).

> Resultado passado não garante resultado futuro. Projeto de estudo, sem dinheiro real.
