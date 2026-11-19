import numpy as np
import pandas as pd


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


FEATURES = ["ret_1d", "ret_2d", "ret_5d", "ret_10d", "sma_ratio_5", "sma_ratio_10",
            "sma_ratio_20", "sma_ratio_50", "rsi_14", "macd", "vol_10d", "vol_30d",
            "hl_range", "volume_chg"]


def compute_features(df):
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 5, 10):
        f[f"ret_{k}d"] = c.pct_change(k)
    for k in (5, 10, 20, 50):
        f[f"sma_ratio_{k}"] = c / c.rolling(k).mean() - 1
    f["rsi_14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    f["macd"] = (ema12 - ema26) / c
    f["vol_10d"] = c.pct_change().rolling(10).std()
    f["vol_30d"] = c.pct_change().rolling(30).std()
    f["hl_range"] = (df["High"] - df["Low"]) / c
    f["volume_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 3)
    return f


def build_features(df):
    f = compute_features(df)
    f["next_ret"] = df["Close"].pct_change().shift(-1)
    f["target"] = (f["next_ret"] > 0).astype(int)
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def latest_features(df):
    row = compute_features(df).replace([np.inf, -np.inf], np.nan).iloc[[-1]][FEATURES]
    if row.isna().any(axis=None):
        raise ValueError("not enough history to compute features")
    return row
