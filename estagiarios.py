"""Os estagiários: cada um é uma estratégia com uma regra fixa.

Cada estagiário devolve uma "posição" para cada candle (1h ou 15min):
1 = comprado (tem a moeda), 0 = fora (está em dólar).
A decisão é tomada com o fechamento do candle e só vale a partir do candle
seguinte, para não "olhar o futuro". Os parâmetros (20, 50, 200...) contam
candles, não dias: em 1h, "Médias 20/50" olha 20 e 50 horas.

Regra para estagiários novos: a posição no candle t só pode usar dados até t.
O arquivo teste_sem_futuro.py confere isso automaticamente.
"""
import pandas as pd


class Estagiario:
    nome = "Estagiário genérico"
    descricao = ""

    def posicoes(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Benchmark(Estagiario):
    """Não é estagiário: é a régua. Compra no 1º dia e nunca mais mexe."""
    nome = "Benchmark (buy and hold)"
    descricao = "Compra e segura. Todo estagiário precisa ganhar dele."

    def posicoes(self, df):
        return pd.Series(1.0, index=df.index)


class CruzamentoMedias(Estagiario):
    def __init__(self, rapida=20, lenta=50):
        self.rapida, self.lenta = rapida, lenta
        self.nome = f"Médias {rapida}/{lenta}"
        self.descricao = "Segue tendência: comprado quando a média rápida está acima da lenta."

    def posicoes(self, df):
        c = df["fechamento"]
        return (c.rolling(self.rapida).mean() > c.rolling(self.lenta).mean()).astype(float)


class ReversaoMedia(Estagiario):
    def __init__(self, janela=20, desvios=2.0):
        self.janela, self.desvios = janela, desvios
        self.nome = f"Reversão {janela}/{desvios}σ"
        self.descricao = "Compra quando o preço cai muito abaixo da média; vende quando volta à média."

    def posicoes(self, df):
        c = df["fechamento"]
        media = c.rolling(self.janela).mean()
        piso = media - self.desvios * c.rolling(self.janela).std()
        sinal = pd.Series(float("nan"), index=df.index)
        sinal[c < piso] = 1.0
        sinal[c > media] = 0.0
        return sinal.ffill().fillna(0.0)


class Breakout(Estagiario):
    def __init__(self, entrada=55, saida=20):
        self.entrada, self.saida = entrada, saida
        self.nome = f"Breakout {entrada}/{saida}"
        self.descricao = "Compra ao romper a máxima recente; sai ao perder a mínima recente."

    def posicoes(self, df):
        c = df["fechamento"]
        maxima = df["maxima"].rolling(self.entrada).max().shift(1)
        minima = df["minima"].rolling(self.saida).min().shift(1)
        sinal = pd.Series(float("nan"), index=df.index)
        sinal[c > maxima] = 1.0
        sinal[c < minima] = 0.0
        return sinal.ffill().fillna(0.0)


class RSI(Estagiario):
    def __init__(self, periodo=14, compra=30, venda=70):
        self.periodo, self.compra, self.venda = periodo, compra, venda
        self.nome = f"RSI {periodo} ({compra}/{venda})"
        self.descricao = "Compra em sobrevenda (RSI baixo); vende em sobrecompra (RSI alto)."

    def posicoes(self, df):
        delta = df["fechamento"].diff()
        ganho = delta.clip(lower=0).ewm(alpha=1 / self.periodo, adjust=False).mean()
        perda = (-delta.clip(upper=0)).ewm(alpha=1 / self.periodo, adjust=False).mean()
        rsi = 100 * ganho / (ganho + perda)  # mesmo RSI, sem dividir por zero
        sinal = pd.Series(float("nan"), index=df.index)
        sinal[rsi < self.compra] = 1.0
        sinal[rsi > self.venda] = 0.0
        return sinal.ffill().fillna(0.0)


def contratar_turma():
    """A turma de estagiários. Adicione ou remova à vontade."""
    return [
        CruzamentoMedias(20, 50),
        CruzamentoMedias(50, 200),
        ReversaoMedia(20, 2.0),
        Breakout(55, 20),
        RSI(14, 30, 70),
    ]
