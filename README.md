# 📈 Market Portfolio Analyzer

A professional-grade portfolio analysis tool built in Python, featuring an interactive
Streamlit dashboard, comprehensive risk metrics, and automated PDF report generation.

---

## 🖥️ Dashboard Preview

> Configure your portfolio in the sidebar, click **Run Analysis**, and explore
> per-ticker technicals alongside portfolio-level risk metrics.

---

## ✨ Features

### Per-Ticker Analysis
- Price chart with Moving Average and Bollinger Bands
- Cumulative Returns and Drawdown visualization
- Rolling Volatility
- Relative Strength Index (RSI)
- Sharpe Ratio, Beta, Alpha, Max Drawdown
- Value at Risk (Historical, Parametric) and CVaR / Expected Shortfall

### Portfolio Analysis
- Multi-ticker returns matrix
- Correlation heatmap
- Weighted portfolio metrics (Return, Volatility, Sharpe, Beta, Alpha)
- Efficient Frontier via Monte Carlo simulation

### Dashboard
- Interactive Plotly charts (zoomable, hoverable)
- Bloomberg-inspired dark theme
- Adjustable tickers, date range, indicator windows and portfolio weights
- Live top movers on idle screen
- CSV and PDF report download

---

## 🗂️ Project Structure
```bash
market-analyzer/
│
├── app.py               # Streamlit dashboard
├── main.py              # Terminal entry point
├── report.py            # PDF report generator
├── config.py            # Constants loader
├── config.yaml          # User configuration
├── requirements.txt
│
├── .streamlit/
│   └── config.toml      # Dashboard theme
│
├── data/
│   ├── loader.py        # download_data, load_data, get_risk_free_rate
│   └── saver.py         # save_data
│
├── analysis/
│   ├── returns.py       # calc_dreturns, calc_cumulative_returns, calc_returns_matrix
│   ├── risk.py          # calc_vol, calc_sharpe_ratio, calc_max_drawdown,
│   │                    # calc_beta, calc_alpha, calc_var_, calc_cvar
│   └── indicators.py    # calc_ma, calc_bollinger_bands, calc_rsi
│
├── portfolio/
│   └── metrics.py       # calc_portfolio_, calc_efficient_frontier
│
└── visualisation/
└── charts.py        # Matplotlib charts for main.py and PDF report
```
---

## ⚙️ Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/market-data-analyzer.git
cd market-data-analyzer
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Streamlit Dashboard
```bash
streamlit run app.py
```

Then in the sidebar:
1. Enter tickers (e.g. `AAPL, MSFT, GOOGL, TSLA`)
2. Set your date range
3. Adjust indicator windows and portfolio weights
4. Click **Run Analysis**

### Terminal
```bash
python main.py
```

Outputs per-ticker CSVs, a portfolio returns CSV, and displays all charts.
Configure tickers, weights, and date range in `config.yaml`.

---

## 🛠️ Configuration

Edit `config.yaml` to set defaults for the terminal entry point:

```yaml
data:
  tickers: ["AAPL", "MSFT", "GOOGL"]
  weights:
    AAPL: 0.5
    MSFT: 0.3
    GOOGL: 0.2
  start_date: "2025-01-01"
  end_date:   "2026-01-01"
  path: "portfolio.csv"

columns:
  date:   "Date"
  open:   "Open"
  high:   "High"
  low:    "Low"
  close:  "Close"
  volume: "Volume"

windows:
  volatility:     20
  moving_average: 20
  rsi:            14

analysis:
  risk_free_rate: 0.043
```

---

## 📦 Dependencies

| Package      | Purpose                          |
|--------------|----------------------------------|
| pandas       | Data manipulation                |
| numpy        | Numerical computation            |
| yfinance     | Market data download             |
| matplotlib   | Charts for terminal and PDF      |
| plotly       | Interactive dashboard charts     |
| streamlit    | Web dashboard framework          |
| reportlab    | PDF report generation            |
| scipy        | Parametric VaR (normal dist)     |
| pyyaml       | Config file parsing              |
| openpyxl     | Excel file support               |

---

## 📐 Metrics Reference

| Metric | Description |
|---|---|
| Sharpe Ratio | Risk-adjusted return (excess return per unit of volatility) |
| Beta | Sensitivity of stock returns relative to the market |
| Alpha | Excess return over what Beta predicts via CAPM |
| Max Drawdown | Largest peak-to-trough loss over the period |
| Rolling Volatility | Standard deviation of returns over a rolling window |
| RSI | Momentum indicator — overbought above 70, oversold below 30 |
| Bollinger Bands | Price bands ±2 std deviations from the moving average |
| Historical VaR | Maximum expected daily loss based on past return distribution |
| Parametric VaR | Maximum expected daily loss assuming normal distribution |
| CVaR | Average loss on days worse than VaR (Expected Shortfall) |
| Efficient Frontier | Risk/return curve from Monte Carlo weight simulation |

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**.
Nothing in this project constitutes financial advice.
Always consult a qualified financial advisor before making investment decisions.

---

## 👤 Author

**Shaun Lee**
[GitHub](https://github.com/shaunljy-creator) · [LinkedIn](https://linkedin.com/in/shaun-lee-1a0707282)