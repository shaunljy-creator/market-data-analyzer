import logging
import os
import pandas as pd

from config         import TICKERS, WINDOW_VOL, WINDOW_MA, CSV_PATH, COL_CLOSE
from data           import download_data, load_data, get_risk_free_rate, save_data
from analysis       import (calc_dreturns, calc_vol, calc_ma, calc_cumulative_returns,
                             calc_max_drawdown, calc_rsi, calc_bollinger_bands,
                             calc_sharpe_ratio, calc_beta, calc_alpha,
                             calc_returns_matrix, calc_correlation_matrix)
from portfolio      import (calc_portfolio_returns, calc_portfolio_volatility,
                             calc_portfolio_sharpe, calc_portfolio_beta,
                             calc_portfolio_alpha, calc_efficient_frontier)
from visualisation  import plot_price, plot_correlation_heatmap, plot_efficient_frontier

logging.basicConfig(level=logging.INFO, filename="app.log", filemode="w",
                    format="%(asctime)s | %(levelname)s | %(message)s")

if __name__ == "__main__":

    # ── Data ─────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        download_data()

    risk_free_rate = get_risk_free_rate()
    df             = load_data()
    benchmark      = download_data(ticker=["^GSPC"], save_path="sp500.csv")
    benchmark      = benchmark.rename(columns={"^GSPC": COL_CLOSE})

    # ── Per-ticker ────────────────────────────────────────
    for ticker in TICKERS:
        ticker_df = pd.DataFrame({COL_CLOSE: df[ticker]})

        ticker_df["Daily Returns"]      = calc_dreturns(ticker_df)
        ticker_df["Volatility"]         = calc_vol(ticker_df, window=WINDOW_VOL)
        ticker_df["MA20"]               = calc_ma(ticker_df, window=WINDOW_MA)
        ticker_df["Cumulative Returns"] = calc_cumulative_returns(ticker_df)
        ticker_df["Drawdown"]           = calc_max_drawdown(ticker_df)
        ticker_df["RSI"]                = calc_rsi(ticker_df)

        bb = calc_bollinger_bands(ticker_df)
        ticker_df["BB_Upper"] = bb["BB_Upper"]
        ticker_df["BB_Lower"] = bb["BB_Lower"]

        sharpe = calc_sharpe_ratio(ticker_df, risk_free_rate=risk_free_rate)
        beta   = calc_beta(ticker_df, benchmark)
        alpha  = calc_alpha(ticker_df, benchmark, risk_free_rate=risk_free_rate)
        max_dd = ticker_df["Drawdown"].min()

        print(f"\n{'=' * 70}")
        print(f"{ticker} Analysis")
        print(f"{'=' * 70}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Beta:         {beta:.4f}")
        print(f"Alpha:        {alpha:.4f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(ticker_df.head())

        plot_price(ticker_df, ticker=ticker)
        save_data(ticker_df, path=f"{ticker}_analysis.csv")

    # ── Portfolio ─────────────────────────────────────────
    returns_matrix = calc_returns_matrix(df)
    corr_matrix    = calc_correlation_matrix(returns_matrix)

    port_returns       = calc_portfolio_returns(returns_matrix)
    port_vol           = calc_portfolio_volatility(returns_matrix)
    port_sharpe        = calc_portfolio_sharpe(returns_matrix, risk_free_rate)
    port_beta          = calc_portfolio_beta(df, benchmark)
    port_alpha         = calc_portfolio_alpha(df, benchmark, risk_free_rate)
    port_return_annual = port_returns.mean() * 252

    print(f"\n{'=' * 70}")
    print("Portfolio Correlation Matrix")
    print(f"{'=' * 70}")
    print(corr_matrix)

    print(f"\n{'=' * 70}")
    print("Portfolio Summary")
    print(f"{'=' * 70}")
    print(f"Annualized Return:     {port_return_annual:.4f}")
    print(f"Annualized Volatility: {port_vol:.4f}")
    print(f"Sharpe Ratio:          {port_sharpe:.4f}")
    print(f"Beta:                  {port_beta:.4f}")
    print(f"Alpha:                 {port_alpha:.4f}")

    frontier = calc_efficient_frontier(returns_matrix, risk_free_rate)

    plot_correlation_heatmap(corr_matrix)
    plot_efficient_frontier(frontier, port_vol, port_return_annual)
    save_data(returns_matrix, path="portfolio_returns.csv")