import pandas as pd
from analysis.returns import calc_dreturns
from data.loader import get_risk_free_rate
import numpy as np
import logging
from config import WINDOW_VOL
from scipy.stats import norm


def calc_vol(df: pd.DataFrame, window: int = WINDOW_VOL) -> pd.Series:
    return calc_dreturns(df).rolling(window=window).std()

def calc_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    returns = calc_dreturns(df).dropna()
    excess  = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std()
    if abs(sharpe) > 3:
        logging.warning(f"Sharpe Ratio abnormally high ({sharpe}) -- Recheck dataset")

    return sharpe

def calc_max_drawdown(df: pd.DataFrame) -> pd.Series:
    cumulative = (1 + calc_dreturns(df)).cumprod()      # portfolio growth curve
    peak       = cumulative.cummax()                    # running peak up to each day
    drawdown   = (cumulative - peak) / peak             # how far below peak we are
    return drawdown

def calc_beta(df: pd.DataFrame, benchmark: pd.DataFrame) -> float:
    # Beta = 1.0  → moves exactly with the market
    # Beta > 1.0  → more volatile than the market(e.g. 1.5 means 50 % more volatile)
    # Beta < 1.0  → less volatile than the market(e.g. 0.5 means half as volatile)
    # Beta < 0    → moves opposite to the market(rare, e.g.gold sometimes)

    # compare returns
    stock_returns = calc_dreturns(df).dropna()
    benchmark_returns = benchmark.iloc[:,0].pct_change().dropna()

    # align both series to the same dates
    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["stock", "benchmark"]

    # beta = covariance(stock, market) / variance(market)
    covariance = aligned["stock"].cov(aligned["benchmark"])
    variance = aligned["benchmark"].var()

    return covariance / variance

def calc_alpha(df: pd.DataFrame,
               benchmark: pd.DataFrame,
               risk_free_rate: int = get_risk_free_rate()) -> float:
    # Alpha > 0  → outperformed the market(good)
    # Alpha = 0  → performed exactly as expected given its risk
    # Alpha < 0  → underperformed the market(bad)

    stock_returns = calc_dreturns(df).dropna().mean() * 252
    benchmark_returns = benchmark.iloc[:,0].pct_change().dropna().mean() * 252

    beta = calc_beta(df, benchmark)

    # CAPM: alpha = actual - (risk free + beta * (market - risk free))
    expected = risk_free_rate + beta * (benchmark_returns - risk_free_rate)
    return stock_returns - expected

def calc_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    # correlation of each ticker's returns against every other
    corr = returns.corr()
    logging.info(f"Calculated correlation matrix")
    return corr

def calc_var_historical(df: pd.DataFrame, confidence: float = 0.95) -> float:
    # looks at actual return distribution — no assumptions about normality
    returns = calc_dreturns(df).dropna()
    var     = returns.quantile(1 - confidence)
    logging.info(f"Historical VaR ({confidence:.0%}): {var:.4f}")
    return var

def calc_var_parametric(df: pd.DataFrame, confidence: float = 0.95) -> float:
    # assumes normal distribution — faster but less accurate for fat tails
    returns = calc_dreturns(df).dropna()
    mean    = returns.mean()
    std     = returns.std()
    var     = mean - norm.ppf(confidence) * std
    logging.info(f"Parametric VaR ({confidence:.0%}): {var:.4f}")
    return var

def calc_cvar(df: pd.DataFrame, confidence: float = 0.95) -> float:
    # CVaR (Conditional VaR) — average loss on the worst days beyond VaR
    # also called "Expected Shortfall" — more informative than VaR alone
    returns   = calc_dreturns(df).dropna()
    var       = calc_var_historical(df, confidence)
    cvar      = returns[returns <= var].mean()  # average of returns worse than VaR
    logging.info(f"CVaR ({confidence:.0%}): {cvar:.4f}")
    return cvar