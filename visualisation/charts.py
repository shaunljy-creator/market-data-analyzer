import matplotlib.pyplot as plt
import pandas as pd
import logging
from config import COL_DATE, COL_CLOSE, WINDOW_VOL, WINDOW_MA
import matplotlib as mpl



def plot_efficient_frontier(frontier: pd.DataFrame,
                             port_vol: float,
                             port_return: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    # scatter all simulated portfolios, coloured by Sharpe ratio
    scatter = ax.scatter(
        frontier["Volatility"],
        frontier["Return"],
        c=frontier["Sharpe"],
        cmap="viridis",
        alpha=0.5,
        s=10
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    # highlight the max Sharpe portfolio
    max_sharpe = frontier.loc[frontier["Sharpe"].idxmax()]
    ax.scatter(max_sharpe["Volatility"], max_sharpe["Return"],
               color="red", marker="*", s=200, label="Max Sharpe")

    # highlight the minimum volatility portfolio
    min_vol = frontier.loc[frontier["Volatility"].idxmin()]
    ax.scatter(min_vol["Volatility"], min_vol["Return"],
               color="blue", marker="*", s=200, label="Min Volatility")

    # plot your actual portfolio
    ax.scatter(port_vol, port_return,
               color="orange", marker="D", s=100, label="Your Portfolio")

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier")
    ax.legend()

    plt.tight_layout()
    plt.show()
    logging.info("Efficient frontier plotted")


# ========================================================
# PLOTTING FUNCTIONS

def plot_price(df: pd.DataFrame, ticker: str="") -> None:
    df = df.reset_index()
    fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
    fig.suptitle(f"{ticker} Stock Analysis", fontsize=16)

    # Plot 1: Price + MA + Bollinger Bands
    axes[0].plot(df[COL_DATE], df[COL_CLOSE], label="Close Price", color="blue")
    axes[0].plot(df[COL_DATE], df["MA20"], label=f"MA{WINDOW_MA}", color="orange", linestyle="--")
    axes[0].plot(df[COL_DATE], df["BB_Upper"], label="Upper", color="grey", linestyle="--", alpha=0.7)
    axes[0].plot(df[COL_DATE], df["BB_Lower"], label="Lower", color="grey", linestyle="--", alpha=0.7)
    axes[0].fill_between(df[COL_DATE], df["BB_Upper"], df["BB_Lower"], color="grey", alpha=0.1)

    axes[0].set_ylabel("Price (USD)")
    axes[0].set_title("Price, Moving Average & Bollinger Bands")
    axes[0].legend()

    # Plot 2: Cumulative Returns & Drawdowns
    drawdown = df["Drawdown"]
    max_dd_val = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    max_dd_date = df[COL_DATE].iloc[max_dd_idx]

    axes[1].plot(df[COL_DATE], df["Cumulative Returns"],  label="Cumulative Returns",   color="green")
    axes[1].fill_between(df[COL_DATE], df["Drawdown"], 0, color="red", alpha=0.3, label="Drawdown")
    axes[1].set_ylabel("Return / Drawdown")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")

    axes[1].scatter(max_dd_date, max_dd_val, color="darkred", zorder=5)
    axes[1].annotate(f"Max Drawdown: {max_dd_val:.2f}",
                     xy=(max_dd_date, max_dd_val),
                     xytext=(max_dd_date, max_dd_val - 0.05),
                     fontsize=9,
                     color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred"))

    axes[1].set_title("Cumulative Returns & Drawdown")
    axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    axes[1].legend()

    # Plot 3: Volatility
    axes[2].plot(df[COL_DATE], df["Volatility"], label=f"Volatility (w={WINDOW_VOL})", color="red")
    axes[2].set_ylabel("Std Dev")
    axes[2].set_title("Rolling Volatility")
    axes[2].legend()

    # Plot 4: RSI
    axes[3].plot(df[COL_DATE], df["RSI"], label="RSI", color="purple")
    axes[3].axhline(70, color="red",   linewidth=0.8, linestyle="--", label="Overbought (70)")
    axes[3].axhline(30, color="green", linewidth=0.8, linestyle="--", label="Oversold (30)")
    axes[3].set_ylabel("RSI")
    axes[3].set_title("Relative Strength Index")
    axes[3].set_ylim(0, 100)   # RSI is always 0-100
    axes[3].set_xlabel("Date")
    axes[3].legend()

    plt.tight_layout()
    plt.show()
    logging.info(f"Plot displayed successfully")


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # manual heatmap using imshow
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label="Correlation")

    # label each cell with its value
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                   ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.index)
    ax.set_title("Portfolio Correlation Matrix")

    plt.tight_layout()
    plt.show()
    logging.info("Correlation heatmap displayed")