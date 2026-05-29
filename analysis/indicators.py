import pandas as pd
from config import COL_CLOSE, WINDOW_MA, WINDOW_RSI


def calc_ma(df: pd.DataFrame, window: int = WINDOW_MA) -> pd.Series:
    return df[COL_CLOSE].rolling(window=window).mean()

def calc_bollinger_bands(df: pd.DataFrame, window: int = WINDOW_MA) -> pd.DataFrame:
    ma = df[COL_CLOSE].rolling(window=window).mean()
    std = df[COL_CLOSE].rolling(window=window).std()
    return pd.DataFrame({
        "BB_Upper": ma + 2 * std,
        "BB_Lower": ma - 2 * std
    })

def calc_rsi(df: pd.DataFrame, window: int = WINDOW_RSI) -> pd.Series:
    # RSI > 70 → overbought(people may start selling soon)
    # RSI < 30 → oversold(people may start buying soon)
    # RSI = 50 → neutral

    delta = df[COL_CLOSE].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs) )

    return rsi
