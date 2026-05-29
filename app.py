import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import yfinance as yf

from config    import COL_CLOSE, COL_DATE
from data      import download_data, get_risk_free_rate
from analysis  import (calc_dreturns, calc_vol, calc_ma, calc_cumulative_returns,
                        calc_max_drawdown, calc_rsi, calc_bollinger_bands,
                        calc_sharpe_ratio, calc_beta, calc_alpha,
                        calc_returns_matrix, calc_correlation_matrix)
from analysis.risk import calc_var_historical, calc_var_parametric, calc_cvar
from portfolio import (calc_portfolio_returns, calc_portfolio_volatility,
                        calc_portfolio_sharpe, calc_portfolio_beta,
                        calc_portfolio_alpha, calc_efficient_frontier)
from report import (generate_report)


# ========================================================
# PAGE CONFIG

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
        /* metric cards */
        [data-testid="metric-container"] {
            background-color: #1A1A1A;
            border: 1px solid #FF6B00;
            border-radius: 4px;
            padding: 12px;
        }
        /* metric value */
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #FF6B00;
            font-family: monospace;
            font-size: 1.4rem;
        }
        /* tab styling */
        .stTabs [data-baseweb="tab"] {
            font-family: monospace;
            color: #E0E0E0;
        }
        .stTabs [aria-selected="true"] {
            color: #FF6B00;
            border-bottom: 2px solid #FF6B00;
        }
        /* sidebar header */
        .css-1d391kg {
            background-color: #1A1A1A;
        }
        /* download button */
        .stDownloadButton button {
            background-color: #FF6B00;
            color: black;
            font-family: monospace;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================================
# SIDEBAR

st.sidebar.title("Portfolio Settings")

# Tickers
raw_tickers = st.sidebar.text_input(
    "Tickers (comma separated)",
    value="AAPL, MSFT, GOOGL"
)
TICKERS = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

# Date range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=date(2025, 1, 1))
end_date   = st.sidebar.date_input("End Date",   value=date(2026, 1, 1))

# Windows
st.sidebar.subheader("Indicator Windows")
window_ma  = st.sidebar.slider("Moving Average Window",  min_value=5,  max_value=100, value=20)
window_vol = st.sidebar.slider("Volatility Window",      min_value=5,  max_value=100, value=20)
window_rsi = st.sidebar.slider("RSI Window",             min_value=5,  max_value=50,  value=14)

# Value at Risk
st.sidebar.subheader("Risk Settings")
confidence = st.sidebar.slider("VaR Confidence Level",
                                min_value=0.90,
                                max_value=0.99,
                                value=0.950,
                                step=0.01)

# Weights
st.sidebar.subheader("Portfolio Weights")
raw_weights = {}
remaining   = 1.0

for i, ticker in enumerate(TICKERS):
    if i < len(TICKERS) - 1:
        w = st.sidebar.slider(
            f"{ticker} Weight",
            min_value=0.0,
            max_value=remaining,
            value=round(remaining / (len(TICKERS) - i), 2),
            step=0.01
        )
        raw_weights[ticker] = w
        remaining -= w
    else:
        # last ticker gets whatever is left so weights always sum to 1
        raw_weights[ticker] = round(remaining, 2)
        st.sidebar.write(f"{ticker} Weight: {round(remaining, 2)}")

WEIGHTS = raw_weights

# ========================================================
# PLOTLY CHART FUNCTIONS

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0A0A0A",
    plot_bgcolor="#0A0A0A",
    font=dict(color="#E0E0E0", family="monospace"),
    xaxis=dict(gridcolor="#2A2A2A", showgrid=True),
    yaxis=dict(gridcolor="#2A2A2A", showgrid=True),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

def plot_ticker(ticker_df: pd.DataFrame, ticker: str, window_vol: int, window_ma: int) -> None:
    ticker_df = ticker_df.reset_index()
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("Price, MA & Bollinger Bands",
                        "Cumulative Returns & Drawdown",
                        "Rolling Volatility",
                        "RSI"),
        vertical_spacing=0.06
    )

    # Plot 1: Price + MA + Bollinger Bands
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df[COL_CLOSE],
                             name="Close Price", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["MA"],
                             name=f"MA{window_ma}", line=dict(color="orange", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["BB_Upper"],
                             name="BB Upper", line=dict(color="grey", dash="dash"), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["BB_Lower"],
                             name="BB Lower", fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
                             line=dict(color="grey", dash="dash"), opacity=0.7), row=1, col=1)

    # Plot 2: Cumulative Returns + Drawdown
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["Cumulative Returns"],
                             name="Cumulative Returns", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["Drawdown"],
                             name="Drawdown", fill="tozeroy", fillcolor="rgba(255,0,0,0.2)",
                             line=dict(color="red")), row=2, col=1)

    # max drawdown annotation
    max_dd_idx  = ticker_df["Drawdown"].idxmin()
    max_dd_date = ticker_df[COL_DATE].iloc[max_dd_idx]
    max_dd_val  = ticker_df["Drawdown"].iloc[max_dd_idx]
    fig.add_annotation(x=max_dd_date, y=max_dd_val,
                       text=f"Max DD: {max_dd_val:.2%}",
                       showarrow=True, arrowhead=1,
                       row=2, col=1)

    # Plot 3: Volatility
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["Volatility"],
                             name=f"Volatility (w={window_vol})", line=dict(color="yellow")), row=3, col=1)

    # Plot 4: RSI
    fig.add_trace(go.Scatter(x=ticker_df[COL_DATE], y=ticker_df["RSI"],
                             name="RSI", line=dict(color="purple")), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold",   row=4, col=1)

    fig.update_layout(height=900, title=f"{ticker} Analysis", showlegend=True)
    fig.update_yaxes(range=[0, 100], row=4, col=1)

    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(corr: pd.DataFrame) -> None:
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Portfolio Correlation Matrix", height=400)
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_frontier(frontier: pd.DataFrame, port_vol: float, port_return: float) -> None:
    fig = go.Figure()

    # all simulated portfolios
    fig.add_trace(go.Scatter(
        x=frontier["Volatility"], y=frontier["Return"],
        mode="markers",
        marker=dict(color=frontier["Sharpe"], colorscale="Viridis",
                    size=4, opacity=0.5, showscale=True,
                    colorbar=dict(title="Sharpe")),
        name="Simulated Portfolios"
    ))

    # max sharpe
    max_sharpe = frontier.loc[frontier["Sharpe"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]],
        mode="markers", marker=dict(color="red", size=14, symbol="star"),
        name="Max Sharpe"
    ))

    # min volatility
    min_vol = frontier.loc[frontier["Volatility"].idxmin()]
    fig.add_trace(go.Scatter(
        x=[min_vol["Volatility"]], y=[min_vol["Return"]],
        mode="markers", marker=dict(color="blue", size=14, symbol="star"),
        name="Min Volatility"
    ))

    # your portfolio
    fig.add_trace(go.Scatter(
        x=[port_vol], y=[port_return],
        mode="markers", marker=dict(color="orange", size=14, symbol="diamond"),
        name="Your Portfolio"
    ))

    fig.update_layout(title="Efficient Frontier", height=500,
                      xaxis_title="Annualized Volatility",
                      yaxis_title="Annualized Return")
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

# ========================================================
# MAIN DASHBOARD

st.title("📈 Portfolio Analyzer")


# shows on the idle screen before user clicks Run
@st.cache_data(ttl=3600)  # cache for 1 hour so it doesn't re-fetch every rerender
def get_top_movers() -> pd.DataFrame:
    # S&P 500 tickers are publicly available
    sp500 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]
    data = yf.download(sp500, period="1d", progress=False)["Close"]
    returns = data.pct_change().iloc[-1].dropna().sort_values(ascending=False)
    return returns


# in the idle screen section
# replace your current run button and idle screen logic with this

# initialise session state on first load
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

if "df" not in st.session_state:
    st.session_state.df = None

# run button sets the flag
if st.sidebar.button("Run Analysis", type="primary", key="btn_run"):
    st.session_state.analysis_run = True

# reset button so user can go back to idle screen
if st.sidebar.button("Reset", type="secondary", key="btn_reset"):
    st.session_state.analysis_run = False
    st.session_state.df = None
    st.rerun()

# idle screen
if not st.session_state.analysis_run:
    st.info("Configure your portfolio in the sidebar and click Run Analysis.")

    st.divider()

    st.subheader("Today's Top Movers")
    movers = get_top_movers()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Gainers**")
        st.dataframe(movers.head(5).apply(lambda x: f"{x:.2%}"))
    with col2:
        st.markdown("**Top Losers**")
        st.dataframe(movers.tail(5).apply(lambda x: f"{x:.2%}"))

    st.divider()

    st.subheader("Common Ticker Reference")
    reference = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "Finance": ["JPM", "GS", "BAC", "V", "MA"],
        "Consumer": ["AMZN", "TSLA", "NKE", "MCD", "SBUX"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    }
    cols = st.columns(len(reference))
    for col, (sector, tickers) in zip(cols, reference.items()):
        col.markdown(f"**{sector}**")
        col.markdown("\n".join(f"• {t}" for t in tickers))

    st.caption("These are common tickers for reference only — not financial recommendations.")

    st.stop()


# ── Data loading ─────────────────────────────────────────
# data loading — only downloads when Run is first clicked
with st.spinner("Downloading data..."):
    try:
        # only download if df not already in session state or tickers changed
        if st.session_state.df is None:
            st.session_state.df        = download_data(
                                            ticker=TICKERS,
                                            start_date=str(start_date),
                                            end_date=str(end_date),
                                            save_path="portfolio.csv"
                                         )
            st.session_state.benchmark = download_data(
                                            ticker=["^GSPC"],
                                            start_date=str(start_date),
                                            end_date=str(end_date),
                                            save_path="sp500.csv"
                                         )
            st.session_state.benchmark = st.session_state.benchmark.rename(
                                            columns={"^GSPC": COL_CLOSE}
                                         )
            st.session_state.risk_free_rate = get_risk_free_rate()

        df             = st.session_state.df
        benchmark      = st.session_state.benchmark
        risk_free_rate = st.session_state.risk_free_rate

    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Per-Ticker Analysis", "Portfolio Analysis"])

# ── Tab 1: Per-ticker ─────────────────────────────────────
with tab1:
    selected = st.selectbox("Select Ticker", TICKERS)

    ticker_df = pd.DataFrame({COL_CLOSE: df[selected]})

    ticker_df["Daily Returns"]      = calc_dreturns(ticker_df)
    ticker_df["Volatility"]         = calc_vol(ticker_df,  window=window_vol)
    ticker_df["MA"]                 = calc_ma(ticker_df,   window=window_ma)
    ticker_df["Cumulative Returns"] = calc_cumulative_returns(ticker_df)
    ticker_df["Drawdown"]           = calc_max_drawdown(ticker_df)
    ticker_df["RSI"]                = calc_rsi(ticker_df,  window=window_rsi)

    bb = calc_bollinger_bands(ticker_df, window=window_ma)
    ticker_df["BB_Upper"] = bb["BB_Upper"]
    ticker_df["BB_Lower"] = bb["BB_Lower"]

    sharpe = calc_sharpe_ratio(ticker_df, risk_free_rate=risk_free_rate)
    beta   = calc_beta(ticker_df, benchmark)
    alpha  = calc_alpha(ticker_df, benchmark, risk_free_rate=risk_free_rate)
    max_dd = ticker_df["Drawdown"].min()

    # metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{sharpe:.4f}")
    col2.metric("Beta",         f"{beta:.4f}")
    col3.metric("Alpha",        f"{alpha:.4f}")
    col4.metric("Max Drawdown", f"{max_dd:.2%}")

    plot_ticker(ticker_df, selected, window_vol, window_ma)

    var_hist = calc_var_historical(ticker_df, confidence)
    var_para = calc_var_parametric(ticker_df, confidence)
    cvar = calc_cvar(ticker_df, confidence)

    # second metrics row
    st.subheader("Risk Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Historical VaR ({confidence:.0%})", f"{var_hist:.2%}",
                help="Maximum expected daily loss based on historical returns")
    col2.metric(f"Parametric VaR ({confidence:.0%})", f"{var_para:.2%}",
                help="Maximum expected daily loss assuming normal distribution")
    col3.metric(f"CVaR ({confidence:.0%})", f"{cvar:.2%}",
                help="Average loss on days worse than VaR — also called Expected Shortfall")

    # CSV download
    st.download_button(
        label="Download CSV",
        data=ticker_df.to_csv(index=True),
        file_name=f"{selected}_analysis.csv",
        mime="text/csv",
        key="dl_ticker_csv"
    )

# ── Tab 2: Portfolio ──────────────────────────────────────
with tab2:
    returns_matrix = calc_returns_matrix(df)
    corr_matrix    = calc_correlation_matrix(returns_matrix)

    port_returns       = calc_portfolio_returns(returns_matrix, weights=WEIGHTS)
    port_vol           = calc_portfolio_volatility(returns_matrix, weights=WEIGHTS)
    port_sharpe        = calc_portfolio_sharpe(returns_matrix, risk_free_rate, weights=WEIGHTS)
    port_beta          = calc_portfolio_beta(df, benchmark, weights=WEIGHTS)
    port_alpha         = calc_portfolio_alpha(df, benchmark, risk_free_rate, weights=WEIGHTS)
    port_return_annual = port_returns.mean() * 252

    # portfolio metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annualized Return",     f"{port_return_annual:.4f}")
    col2.metric("Annualized Volatility", f"{port_vol:.4f}")
    col3.metric("Sharpe Ratio",          f"{port_sharpe:.4f}")
    col4.metric("Beta",                  f"{port_beta:.4f}")
    col5.metric("Alpha",                 f"{port_alpha:.4f}")

    plot_heatmap(corr_matrix)

    with st.spinner("Simulating efficient frontier..."):
        frontier = calc_efficient_frontier(returns_matrix, risk_free_rate)
    plot_frontier(frontier, port_vol, port_return_annual)

    # portfolio CSV download
    st.download_button(
        label="Download Portfolio Returns CSV",
        data=returns_matrix.to_csv(index=True),
        file_name="portfolio_returns.csv",
        mime="text/csv",
        key="dl_portfolio_csv"
    )

    # PDF DOWNLOADER
    # build inputs for the report
    ticker_data = {}
    ticker_metrics = {}

    for ticker in TICKERS:
        t_df = pd.DataFrame({COL_CLOSE: df[ticker]})
        t_df["Daily Returns"] = calc_dreturns(t_df)
        t_df["Volatility"] = calc_vol(t_df, window=window_vol)
        t_df["MA"] = calc_ma(t_df, window=window_ma)
        t_df["Cumulative Returns"] = calc_cumulative_returns(t_df)
        t_df["Drawdown"] = calc_max_drawdown(t_df)
        t_df["RSI"] = calc_rsi(t_df, window=window_rsi)
        bb = calc_bollinger_bands(t_df, window=window_ma)
        t_df["BB_Upper"] = bb["BB_Upper"]
        t_df["BB_Lower"] = bb["BB_Lower"]

        ticker_data[ticker] = t_df
        ticker_metrics[ticker] = {
            "sharpe": calc_sharpe_ratio(t_df, risk_free_rate=risk_free_rate),
            "beta": calc_beta(t_df, benchmark),
            "alpha": calc_alpha(t_df, benchmark, risk_free_rate=risk_free_rate),
            "max_dd": t_df["Drawdown"].min(),
            "var_hist": calc_var_historical(t_df, confidence),
            "var_para": calc_var_parametric(t_df, confidence),
            "cvar": calc_cvar(t_df, confidence),
        }

    port_metrics = {
        "return": port_return_annual,
        "vol": port_vol,
        "sharpe": port_sharpe,
        "beta": port_beta,
        "alpha": port_alpha,
    }

    # PDF download button
    with st.spinner("Generating report..."):
        pdf_bytes = generate_report(
            ticker_data=ticker_data,
            ticker_metrics=ticker_metrics,
            returns_matrix=returns_matrix,
            corr_matrix=corr_matrix,
            frontier=frontier,
            port_metrics=port_metrics,
            port_vol=port_vol,
            port_return=port_return_annual,
            window_ma=window_ma,
            window_vol=window_vol,
        )

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"portfolio_report_{date.today()}.pdf",
        mime="application/pdf",
        key="dl_pdf"
    )