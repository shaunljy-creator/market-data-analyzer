import pandas as pd
import logging

# DATA SAVING

def save_data(df: pd.DataFrame, path: str = "apple_stocks_processed.csv") -> None:
    df.to_csv(path, index=True)
    logging.info(f"Data saved to {path}")