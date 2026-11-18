# Empresa de Trading (simulação)

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
