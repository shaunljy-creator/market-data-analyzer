import pandas as pd
import logging
import numpy as np
from analysis.risk import calc_beta, calc_alpha
from config import COL_CLOSE, WEIGHTS



def calc_portfolio_returns(returns_matrix: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    if weights is None:
        weights = WEIGHTS
    w = np.array([weights[t] for t in returns_matrix.columns])
    # dot product of daily returns x weights = single portfolio return per day
    return returns_matrix.dot(w)

def calc_portfolio_volatility(returns_matrix: pd.DataFrame, weights: dict = None) -> float:
    if weights is None:
        weights = WEIGHTS
    w = np.array([weights[t] for t in returns_matrix.columns])
    cov_matrix = returns_matrix.cov() * 252  # annualized covariance
    return np.sqrt(w.T @ cov_matrix @ w) # w^t * X * w

def calc_portfolio_sharpe(returns_matrix: pd.DataFrame, risk_free_rate: float = 0.0, weights: dict = None) -> float:
    if weights is None:
        weights = WEIGHTS
    port_returns = calc_portfolio_returns(returns_matrix, weights)
    excess = port_returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std()

def calc_portfolio_beta(df: pd.DataFrame, benchmark: pd.DataFrame, weights: dict = None) -> float:
    if weights is None:
        weights = WEIGHTS
    weighted_beta = 0.0
    for ticker in weights:
        ticker_df = pd.DataFrame({COL_CLOSE: df[ticker]})  # ← df, not returns_matrix
        weighted_beta += weights[ticker] * calc_beta(ticker_df, benchmark)
    logging.info(f"Portfolio Beta: {weighted_beta:.4f}")
    return weighted_beta

def calc_portfolio_alpha(df: pd.DataFrame, benchmark: pd.DataFrame, risk_free_rate: float = 0.0, weights: dict = None) -> float:
    if weights is None:
        weights = WEIGHTS
    weighted_alpha = 0.0
    for ticker in weights:
        ticker_df = pd.DataFrame({COL_CLOSE: df[ticker]})  # ← df, not returns_matrix
        weighted_alpha += weights[ticker] * calc_alpha(ticker_df, benchmark, risk_free_rate)
    logging.info(f"Portfolio Alpha: {weighted_alpha:.4f}")
    return weighted_alpha

def calc_efficient_frontier(returns_matrix: pd.DataFrame,
                             risk_free_rate: float = 0.0,
                             simulations: int = 5000) -> pd.DataFrame:

    n_assets  = len(returns_matrix.columns)
    results   = []

    for _ in range(simulations):
        # generate random weights that sum to 1
        weights = np.random.dirichlet(np.ones(n_assets))

        port_return = (returns_matrix.mean().dot(weights)) * 252  # annualized
        port_vol    = np.sqrt(weights.T @ (returns_matrix.cov() * 252) @ weights)
        sharpe      = (port_return - risk_free_rate) / port_vol

        results.append({
            "Return":     port_return,
            "Volatility": port_vol,
            "Sharpe":     sharpe,
            "Weights":    weights
        })

    logging.info(f"Efficient frontier simulated with {simulations} portfolios")
    return pd.DataFrame(results)