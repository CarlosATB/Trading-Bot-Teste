# Empresa de Trading (simulação)

> **[Segundo resultado (v0.2)](RESULTADOS_v0.2.md):** nenhum bot aprovado ainda; a Luna terminou a prova com diferença média positiva contra o BTC e passou a capturar 29% das altas. [Primeiro resultado (v0.1)](RESULTADOS.md).

Laboratório para testar estratégias de trade com dados históricos, organizado como uma empresa:

- **Dono:** você. Define as regras do RH e decide o que fazer com os resultados.
- **Gerente (RH):** código em `empresa.py`. Avalia os estagiários e redistribui o capital.
- **Estagiários:** as estratégias em `estagiarios.py`.
- **Benchmark:** o "buy and hold". Não é estagiário, é a régua que todos precisam superar.

Nada aqui usa dinheiro real.

## Como rodar

```bash
pip install -r requirements.txt
python teste_sem_futuro.py                 # confere que ninguém olha o futuro
python main.py --sintetico                 # teste sem internet
python main.py                             # BTC/USDT real, candles de 1h e 15min
python main.py --simbolo ETHUSDT --intervalos 1h
```

Os dados vêm da API pública da Binance (não precisa de conta) e ficam salvos em `cache/`.
As tabelas de resultado ficam em `relatorios/`.

## Como funciona o teste semana a semana

- **Cada semana é um teste separado** (segunda 00:00 a domingo 23:59, UTC). Todos começam em dólar e terminam em dólar.
- **Candles de 1h** (168 por semana) são o padrão; **15min** (672 por semana) roda como comparação.
- **Indicadores aquecidos:** são calculados com o histórico anterior à semana (os dados começam em set/2020), mas cada decisão usa só dados até o candle em que foi tomada. A decisão vale a partir do candle seguinte.
- **Custos:** cada troca de posição custa 0,15% (taxa 0,1% + slippage 0,05%). Ficar parado não custa nada. O Benchmark compra no início e vende no fim de cada semana, então paga 2 trocas.
- **Treino (2021 a 2023):** as 156 semanas em sequência, com o RH avaliando.
- **Prova (2024 e 2025):** 20 semanas sorteadas uma única vez e guardadas em `relatorios/semanas_sorteadas.csv`. As rodadas seguintes reusam o mesmo sorteio. **Aprovado** = sobreviveu ao treino e ganhou do Benchmark em mais da metade das semanas (11 ou mais de 20).
- A coluna **"Chance de ser sorte"** mostra a chance de um bot qualquer, jogando cara ou coroa, ganhar aquele número de semanas ou mais. Com 11 de 20 ela é 41%: passar raspando não prova nada.

## Fase 1 de Machine Learning (`main_ml.py`)

Os estagiários de regras fixas viraram "veteranos" (`main.py`). A turma nova aprende sozinha:

| Estagiário | Modelo |
|---|---|
| **Xavier** | XGBoost (na GPU) |
| **Luna** | LightGBM + meta-labeling (um segundo modelo aprende com as entradas que ela propôs e veta as ruins) |
| **Régua** | Regressão logística. Não é estagiário: se um modelo complexo não ganha dela, está decorando o passado |

Fases 2 e 3 (planejadas): Lucas (LSTM), Tereza (Transformer temporal) e Paulo (Reinforcement Learning, PPO).

**O que eles aprendem:** rótulos *triple-barrier*. Para cada hora, o preço bate primeiro a barreira de lucro, a de prejuízo ou a de tempo (24h)? As barreiras medem 1 desvio padrão esperado em 24h. O modelo devolve as três probabilidades.

**Uma regra para todos** (`ml/regra.py`): ninguém opera só porque viu uma tendência. A entrada (compra no spot ou venda no perpétuo) só acontece se, juntando a probabilidade de lucro, a de prejuízo, o tempo esperado até as barreiras, as taxas, o spread e o slippage e o funding, o valor esperado for positivo e a relação retorno/risco passar do mínimo. O tamanho vem de meio Kelly, limitado pela volatilidade.

**Features** (`ml/features.py`): cerca de 60, todas estacionárias. Retornos em várias escalas, volatilidade realizada e de Parkinson, fluxo agressor, número de negócios, ETH como ativo cruzado, basis spot-perpétuo, funding, open interest e razões long/short dos futuros, além de hora e dia.

**Como aprendem sem olhar o futuro** (`ml/walkforward.py`):
- *Estudo inicial:* antes de operar, aprendem com todo o histórico desde 2019 e passam a primeira semana só observando.
- *Walk-forward semanal:* toda semana são retreinados só com rótulos que terminaram antes da semana operada (purging).
- *Aprender com os erros:* os candles em que erraram ao vivo ganham mais peso no treino seguinte.
- *Pesos:* unicidade dos rótulos sobrepostos (López de Prado) e recência, com meia-vida de 1 ano.

**Etapas:** 2020 é o estágio, que não conta para o RH. 2021 a 2023 é o treino, avaliado pelo RH. A prova são as mesmas 20 semanas sorteadas de 2024 e 2025.

**Rigor:** o `otimizar.py` (Optuna) usa só dados até 2023, com validação purgada em ordem no tempo. O Deflated Sharpe (DSR) desconta todas as tentativas feitas. Cada vez que a prova roda fica registrada em `relatorios/registro_da_prova.csv`. Os experimentos ficam no MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

```bash
python teste_sem_futuro.py      # confere que nada olha o futuro
python otimizar.py --tentativas 30
python main_ml.py               # estágio + treino
python main_ml.py --prova       # + a prova (rode o mínimo de vezes)
```

## Plano de carreira (em `RegrasRH`)

| Situação | Consequência |
|---|---|
| Primeiras 26 semanas (~6 meses) | Período de experiência: todos recebem o mesmo capital |
| Avaliação | O RH avalia a cada 4 semanas (~1 mês) |
| Ganhou do Benchmark por mais de 5 p.p. nas últimas 26 semanas | **Promovido**: dobro de capital |
| Perdeu do Benchmark nas últimas 26 semanas | **Advertido**: metade do capital |
| 3 advertências seguidas | **Demitido**: capital zero |
| Queda máxima acima de 50% | **Demitido** na hora |

## Como contratar um estagiário novo

Crie uma classe em `estagiarios.py` com `nome` e um método `posicoes(df)` que devolva 1 (comprado) ou 0 (fora) para cada candle, usando só dados até aquele candle. Adicione ela em `contratar_turma()` e rode `python teste_sem_futuro.py`.

## Aviso

Resultado passado não garante resultado futuro. É um projeto de estudo, não um sinal para investir.
