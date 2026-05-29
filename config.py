import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ====================================================================

CSV_PATH =   config["data"]["path"]
COL_DATE =   config["columns"]["date"]
COL_OPEN =   config["columns"]["open"]
COL_HIGH =   config["columns"]["high"]
COL_LOW =    config["columns"]["low"]
COL_CLOSE =  config["columns"]["close"]
COL_VOLUME = config["columns"]["volume"]
WINDOW_VOL = config["windows"]["volatility"]
WINDOW_MA =  config["windows"]["moving_average"]
WINDOW_RSI = config["windows"]["rsi"]
TICKERS = config["data"]["tickers"]
WEIGHTS = config["data"]["weights"]
START_DATE = config["data"]["start_date"]
END_DATE = config["data"]["end_date"]

# ====================================================================

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6