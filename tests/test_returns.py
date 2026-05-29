import pandas as pd
import numpy as np
import pytest
from main import calc_dreturns, calc_vol, calc_sharpe_ratio

# Test Case
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "Close": [100.0, 102.0, 101.0, 105.0, 103.0],
        "Open": [99.0, 101.0, 100.0, 104.0, 102.0],
        "High": [103.0, 104.0, 103.0, 107.0, 105.0],
        "Low": [98.0, 100.0, 99.0, 103.0, 101.0],
        "Volume": [1000, 1100, 900, 1200, 1050],
    })

    def test_dreturns_length(sample_df):
        result = calc_dreturns(sample_df)
        assert len(result) == len(sample_df)  # output should be same length as input

    def test_dreturns_first_is_nan(sample_df):
        result = calc_dreturns(sample_df)
        assert np.isnan(result.iloc[0])  # first value is always NaN with pct_change()

    def test_dreturns_values(sample_df):
        result = calc_dreturns(sample_df)
        assert round(result.iloc[1], 4) == 0.02  # 102/100 - 1 = 0.02

    def test_vol_respects_window(sample_df):
        result = calc_vol(sample_df, window=3)
        assert result.iloc[1] is np.nan or np.isnan(result.iloc[1])  # not enough data yet

    def test_sharpe_returns_float(sample_df):
        result = calc_sharpe_ratio(sample_df)
        assert isinstance(result, float)