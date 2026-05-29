import logging
import pandas as pd
import yfinance as yf
from config import CSV_PATH, TICKERS, WINDOW_VOL, WINDOW_MA, START_DATE, END_DATE


# NORMALIZE DATA

def normalize_columns(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    # if yfinance returned a MultiIndex (happens with newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        logging.info("MultiIndex columns detected — flattening")
        # filter to just this ticker's columns, then drop the ticker level
        if tickers and len(tickers) == 1:
            df = df.xs(tickers[0], axis=1, level="Ticker")
        else:
            # for multi-ticker, just drop the Price level and keep Ticker as columns
            df = df["Close"] if "Close" in df.columns.get_level_values(0) else df

    # only apply title case if columns are plain strings
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [col.strip().title() for col in df.columns]

    # rename any known yfinance variants to your standard names
    rename_map = {"Adj Close": "Close"}
    drop_cols = {"Stock Splits", "Dividends"}

    # drop unwanted columns
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # apply renames
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    logging.info(f"Normalized columns: {list(df.columns)}")
    return df


# DOWNLOAD DATA

def download_data(
        ticker: list[str] = TICKERS,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        save_path: str = CSV_PATH,
    ) -> pd.DataFrame:

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        logging.error(f"No data for {ticker}")
        raise ValueError("No data downloaded.")

    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.to_csv(save_path, index=True)
    logging.info(f"Downloaded {len(df)} rows of data from {ticker} to {save_path}")

    return df


# LOAD DATA

def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logging.info(f"Successfully loaded {len(df)} rows from {path}")
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"File {path} not found.")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise RuntimeError(f"Failed to read file: {e}.")

    missing  = [c for c in TICKERS if c not in df.columns]
    if missing:
        logging.error(f"Missing tickers: {missing}")
        raise ValueError(f"Missing tickers: {missing}\n")

    df.sort_index(inplace=True)

    # WARNING FOR SMALL DATASET WRT WINDOWS
    if len(df) < max(WINDOW_VOL, WINDOW_MA):
        logging.warning(f"Dataset has {len(df)} rows -- window sizes ({WINDOW_VOL}, {WINDOW_MA}) may produce mostly NaNs")

    logging.info(f"Successfully loaded {len(df)} rows from {path}")
    return df

# RISK FREE RATE GETTER
def get_risk_free_rate() -> float:
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)
        if tnx.empty:
            logging.warning("Could not fetch risk-free rate, defaulting to 0.0 instead")
            return 0.0

        if isinstance(tnx.columns, pd.MultiIndex):
            tnx = tnx["Close"].iloc[:,0]
        else:
            tnx = tnx["Close"]

        rate = float(tnx.dropna().iloc[-1]) / 100       # returns as a percentage
        logging.info(f"Risk-free rate fetched: {rate:.4f}")
        return rate

    except Exception as e:
        logging.warning(f"Failed to fetch risk-free rate: {e} -- Defaulting to 0.0 instead")
        return 0.0
