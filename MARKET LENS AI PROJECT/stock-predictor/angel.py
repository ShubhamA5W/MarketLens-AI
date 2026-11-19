import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

TOKENS = {
    "^NSEI": ("NSE", "99926000"),
    "^BSESN": ("BSE", "99919000"),
    "RELIANCE.NS": ("NSE", "2885"),
    "TCS.NS": ("NSE", "11536"),
    "INFY.NS": ("NSE", "1594"),
    "HDFCBANK.NS": ("NSE", "1333"),
}
REQUIRED = ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET")
IST = timedelta(hours=5, minutes=30)
_daily_cache = {}


def configured(ticker):
    return ticker.upper() in TOKENS and all(os.environ.get(k) for k in REQUIRED)


def _ist_now():
    return datetime.now(timezone.utc).replace(tzinfo=None) + IST


class _Session:
    def __init__(self):
        self.api = None
        self.started = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self.api is None or time.time() - self.started > 6 * 3600:
                import pyotp
                from SmartApi import SmartConnect
                api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
                totp = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
                resp = api.generateSession(os.environ["ANGEL_CLIENT_ID"], os.environ["ANGEL_PIN"], totp)
                if not resp or not resp.get("status"):
                    raise RuntimeError("Angel One login failed, check your .env values")
                self.api, self.started = api, time.time()
            return self.api

    def reset(self):
        with self.lock:
            self.api = None


_session = _Session()


def _candles(api, exchange, token, interval, start, end):
    resp = api.getCandleData({
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": start.strftime("%Y-%m-%d %H:%M"),
        "todate": end.strftime("%Y-%m-%d %H:%M"),
    })
    rows = (resp or {}).get("data") or []
    if not rows:
        raise ValueError("no candles returned")
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    ts = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None) + pd.Timedelta(hours=5, minutes=30)
    df = df.drop(columns="ts").astype(float)
    df.index = pd.DatetimeIndex(ts)
    return df.sort_index()


def _daily(api, key, exchange, token):
    hit = _daily_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    now = _ist_now()
    df = _candles(api, exchange, token, "ONE_DAY", now - timedelta(days=400), now)
    df.index = df.index.normalize()
    _daily_cache[key] = (time.time(), df)
    return df


def _ltp(api, exchange, token):
    try:
        resp = api.getMarketData("LTP", {exchange: [token]})
        return float(resp["data"]["fetched"][0]["ltp"])
    except Exception:
        return None


def snapshot(ticker):
    key = ticker.upper()
    exchange, token = TOKENS[key]
    try:
        api = _session.get()
        now = _ist_now()
        daily = _daily(api, key, exchange, token)
        intra = _candles(api, exchange, token, "ONE_MINUTE", (now - timedelta(days=5)).replace(hour=9, minute=15), now)
        ltp = _ltp(api, exchange, token)
    except Exception:
        _session.reset()
        raise

    day = intra.index[-1].normalize()
    intra = intra[intra.index.normalize() == day]
    age = (now - intra.index[-1].to_pydatetime()).total_seconds()
    market_open = age < 4 * 60 and day == pd.Timestamp(now).normalize()
    price = ltp if ltp else float(intra["Close"].iloc[-1])

    times = [t.strftime("%H:%M") for t in intra.index]
    prices = [float(v) for v in intra["Close"]]
    if market_open:
        times.append(now.strftime("%H:%M:%S"))
        prices.append(price)

    hist = daily[daily.index < day]
    bar = pd.DataFrame({
        "Open": [float(intra["Open"].iloc[0])],
        "High": [max(float(intra["High"].max()), price)],
        "Low": [min(float(intra["Low"].min()), price)],
        "Close": [price],
        "Volume": [float(intra["Volume"].sum())],
    }, index=[day])
    return {"daily": pd.concat([hist, bar]), "prev_close": float(hist["Close"].iloc[-1]), "price": price,
            "times": times, "prices": prices, "market_open": market_open, "synthetic": False,
            "source": "Angel One"}


_IV = {"1m": ("ONE_MINUTE", 3), "5m": ("FIVE_MINUTE", 10), "15m": ("FIFTEEN_MINUTE", 20),
       "1h": ("ONE_HOUR", 60), "1d": ("ONE_DAY", 200)}


def _market_hours(now):
    mins = now.hour * 60 + now.minute
    return now.weekday() < 5 and 9 * 60 + 15 <= mins <= 15 * 60 + 30


def candles(ticker, interval):
    exchange, token = TOKENS[ticker.upper()]
    name, days = _IV[interval]
    try:
        api = _session.get()
        now = _ist_now()
        df = _candles(api, exchange, token, name, (now - timedelta(days=days)).replace(hour=9, minute=15), now)
        ltp = _ltp(api, exchange, token)
    except Exception:
        _session.reset()
        raise
    if ltp and _market_hours(now) and df.index[-1].normalize() == pd.Timestamp(now).normalize():
        df.iloc[-1, df.columns.get_loc("Close")] = ltp
        df.iloc[-1, df.columns.get_loc("High")] = max(df["High"].iloc[-1], ltp)
        df.iloc[-1, df.columns.get_loc("Low")] = min(df["Low"].iloc[-1], ltp)
    return df
