import numpy as np
import pandas as pd

def synthetic_prices(n=2500, seed=7):
    rng = np.random.default_rng(seed)
    vol = np.empty(n); vol[0] = 0.01
    ret = np.empty(n)
    for t in range(n):
        if t:
            vol[t] = np.sqrt(1e-6 + 0.08 * ret[t-1] ** 2 + 0.9 * vol[t-1] ** 2)
        ret[t] = rng.normal(0.0003, vol[t])
    close = 1000 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"Close": close}, index=idx)
    df["Open"] = df.Close.shift(1).fillna(df.Close.iloc[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + rng.uniform(0, 0.01, n))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - rng.uniform(0, 0.01, n))
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    return df

def load_prices(ticker="^NSEI", start="2012-01-01"):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 500:
            raise ValueError("too little data")
        print(f"[data] {ticker}: {len(df)} rows from Yahoo Finance")
        return df, ticker
    except Exception as e:
        print(f"[data] Yahoo Finance unavailable ({type(e).__name__}); using SYNTHETIC data")
        return synthetic_prices(), "SYNTHETIC"


import time as _time

_sim = {}


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _naive(idx):
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def _bar(o, h, l, c, v, day):
    return pd.DataFrame({"Open": [o], "High": [h], "Low": [l], "Close": [c], "Volume": [v]}, index=[day])


def _simulated(ticker):
    s = _sim.get(ticker)
    if s is None:
        base = synthetic_prices()
        s = {"base": base, "rng": np.random.default_rng(), "p": [float(base["Close"].iloc[-1])], "t": [_time.strftime("%H:%M:%S")]}
        _sim[ticker] = s
    s["p"].append(s["p"][-1] * float(np.exp(s["rng"].normal(0, 0.0006))))
    s["t"].append(_time.strftime("%H:%M:%S"))
    s["p"], s["t"] = s["p"][-400:], s["t"][-400:]
    base = s["base"]
    day = base.index[-1] + pd.offsets.BDay(1)
    full = pd.concat([base, _bar(s["p"][0], max(s["p"]), min(s["p"]), s["p"][-1], 2_000_000, day)])
    return {"daily": full, "prev_close": float(base["Close"].iloc[-1]), "price": s["p"][-1],
            "times": list(s["t"]), "prices": list(s["p"]), "market_open": True, "synthetic": True, "source": "Simulated"}


def live_snapshot(ticker, synthetic=False):
    if synthetic:
        return _simulated(ticker)
    try:
        import angel
        if angel.configured(ticker):
            return angel.snapshot(ticker)
    except Exception as e:
        print(f"[live] Angel One failed ({type(e).__name__}: {e}); falling back to Yahoo Finance")
    import yfinance as yf
    cols = ["Open", "High", "Low", "Close", "Volume"]
    daily = _flatten(yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False))[cols].dropna()
    intra = _flatten(yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False))[cols].dropna()
    if len(daily) < 80 or intra.empty:
        raise ValueError("no live data returned")
    market_open = False
    if getattr(intra.index, "tz", None) is not None:
        age = (pd.Timestamp.now(tz="UTC") - intra.index[-1].tz_convert("UTC")).total_seconds()
        market_open = age < 25 * 60
    intra.index = _naive(intra.index)
    daily.index = _naive(daily.index)
    day = intra.index[-1].normalize()
    hist = daily[daily.index.normalize() < day]
    bar = _bar(intra["Open"].iloc[0], intra["High"].max(), intra["Low"].min(),
               intra["Close"].iloc[-1], intra["Volume"].sum(), day)
    return {"daily": pd.concat([hist, bar]), "prev_close": float(hist["Close"].iloc[-1]),
            "price": float(intra["Close"].iloc[-1]), "times": [t.strftime("%H:%M") for t in intra.index],
            "prices": [float(v) for v in intra["Close"]], "market_open": market_open, "synthetic": False, "source": "Yahoo Finance"}


import zlib

_RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}
_YF_PERIOD = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "6mo"}
_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


def _label(idx, interval):
    if interval == "1d":
        fmt = "%d %b"
    elif idx[-1].normalize() != idx[0].normalize():
        fmt = "%d %H:%M"
    else:
        fmt = "%H:%M"
    return [t.strftime(fmt) for t in idx]


def to_candles(df, interval, n=120):
    df = df.tail(n)
    labels = _label(df.index, interval)
    return [{"t": labels[i], "o": round(float(r.Open), 4), "h": round(float(r.High), 4),
             "l": round(float(r.Low), 4), "c": round(float(r.Close), 4), "v": float(r.Volume)}
            for i, r in enumerate(df.itertuples())]


def _sim_candles(ticker, interval, n=120):
    if ticker not in _sim:
        _simulated(ticker)
    s = _sim[ticker]
    price = s["p"][-1]
    if interval == "1d":
        base = s["base"][["Open", "High", "Low", "Close", "Volume"]].tail(n - 1)
        day = base.index[-1] + pd.offsets.BDay(1)
        return pd.concat([base, _bar(s["p"][0], max(s["p"]), min(s["p"]), price, 2_000_000, day)])
    step = _MINUTES[interval]
    total = n * step + step
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    path = float(s["base"]["Close"].iloc[-1]) * np.exp(np.cumsum(rng.normal(0, 0.0005, total)))
    path = path * (float(s["base"]["Close"].iloc[-1]) / path[-1])
    openp = np.concatenate([[path[0]], path[:-1]])
    wick = rng.uniform(0, 0.0003, total)
    now = pd.Timestamp.now().floor("min")
    idx = pd.date_range(end=now, periods=total, freq="1min")
    m = pd.DataFrame({"Open": openp, "High": np.maximum(openp, path) * (1 + wick),
                      "Low": np.minimum(openp, path) * (1 - wick), "Close": path,
                      "Volume": rng.integers(1000, 9000, total)}, index=idx)
    out = m.resample(_RULE[interval]).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    out.iloc[-1, out.columns.get_loc("Close")] = price
    out.iloc[-1, out.columns.get_loc("High")] = max(out["High"].iloc[-1], price)
    out.iloc[-1, out.columns.get_loc("Low")] = min(out["Low"].iloc[-1], price)
    return out


def _yahoo_candles(ticker, interval):
    import yfinance as yf
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = _flatten(yf.download(ticker, period=_YF_PERIOD[interval], interval=interval,
                              auto_adjust=True, progress=False))[cols].dropna()
    if df.empty:
        raise ValueError("no candle data returned")
    df.index = _naive(df.index)
    return df


def live_candles(ticker, interval, synthetic=False):
    if interval not in _RULE:
        raise ValueError("unsupported interval")
    if synthetic:
        return to_candles(_sim_candles(ticker, interval), interval), "Simulated"
    try:
        import angel
        if angel.configured(ticker):
            return to_candles(angel.candles(ticker, interval), interval), "Angel One"
    except Exception as e:
        print(f"[candles] Angel One failed ({type(e).__name__}: {e}); falling back to Yahoo Finance")
    return to_candles(_yahoo_candles(ticker, interval), interval), "Yahoo Finance"
