import pandas as pd
import logging
from config import COL_CLOSE

def calc_dreturns(df: pd.DataFrame) -> pd.Series:
    return df[COL_CLOSE].pct_change()

def calc_cumulative_returns(df: pd.DataFrame) -> pd.Series:
    return (1 + calc_dreturns(df)).cumprod() - 1

def calc_returns_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # each column becomes its daily % returns
    returns = df.pct_change().dropna()
    logging.info(f"Calculated returns matrix: {returns.shape}")
    return returns