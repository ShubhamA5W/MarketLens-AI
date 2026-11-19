import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import live_candles, live_snapshot, load_prices
from features import FEATURES, build_features, latest_features

COST = 0.001
_cache = {}
_best = {}
_live_cache = {}
_candle_cache = {}


def _models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
    }


def _stats(curve):
    r = curve.pct_change().dropna()
    yrs = max(len(curve) / 252, 1e-9)
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0
    dd = (curve / curve.cummax() - 1).min()
    return {"cagr": float(cagr), "sharpe": float(sharpe), "max_dd": float(dd), "total": float(curve.iloc[-1] - 1)}


def _thin(s, n=220):
    step = max(len(s) // n, 1)
    return [round(float(v), 4) for v in s.iloc[::step]]


def analyze(ticker):
    key = ticker.upper()
    if key in _cache:
        return _cache[key]

    df, name = load_prices(ticker)
    feat = build_features(df)
    X, y = feat[FEATURES], feat["target"]
    split = int(len(feat) * 0.8)
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    rows, probs, fitted = [], {}, {}
    for n, m in _models().items():
        cv = [accuracy_score(y_tr.iloc[v], m.fit(X_tr.iloc[t], y_tr.iloc[t]).predict(X_tr.iloc[v]))
              for t, v in TimeSeriesSplit(5).split(X_tr)]
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        probs[n], fitted[n] = p, m
        rows.append({"model": n, "cv_acc": float(np.mean(cv)), "acc": float(accuracy_score(y_te, p > 0.5)),
                     "auc": float(roc_auc_score(y_te, p))})

    best = max(rows, key=lambda r: r["auc"])["model"]
    pos = pd.Series((probs[best] > 0.5).astype(int), index=X_te.index)
    gross = pos * feat.loc[X_te.index, "next_ret"]
    strat = (1 + gross - pos.diff().abs().fillna(0) * COST).cumprod()
    bh = (1 + feat.loc[X_te.index, "next_ret"]).cumprod()

    rf = fitted["Random Forest"]
    imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])[:8]

    _best[key] = fitted[best]
    latest = latest_features(df)
    p_up = float(fitted[best].predict_proba(latest)[0, 1])
    close = df["Close"]

    out = {
        "ticker": name,
        "synthetic": name == "SYNTHETIC",
        "as_of": str(latest.index[0].date()),
        "last_close": float(close.iloc[-1]),
        "day_change": float(close.iloc[-1] / close.iloc[-2] - 1),
        "p_up": p_up,
        "signal": "UP" if p_up > 0.5 else "DOWN",
        "best_model": best,
        "baseline": float(max(y_te.mean(), 1 - y_te.mean())),
        "train_period": [str(X_tr.index[0].date()), str(X_tr.index[-1].date())],
        "test_period": [str(X_te.index[0].date()), str(X_te.index[-1].date())],
        "models": rows,
        "strategy": _stats(strat),
        "buy_hold": _stats(bh),
        "curve_strategy": _thin(strat),
        "curve_bh": _thin(bh),
        "importance": [{"name": n, "value": float(v)} for n, v in imp],
    }
    _cache[key] = out
    return out


def _thin_pair(times, prices, n=240):
    step = max(len(prices) // n, 1)
    idx = list(range(0, len(prices), step))
    if idx[-1] != len(prices) - 1:
        idx.append(len(prices) - 1)
    return [times[i] for i in idx], [round(prices[i], 4) for i in idx]


def live(ticker):
    key = ticker.upper()
    hit = _live_cache.get(key)
    if hit and time.time() - hit[0] < 4:
        return hit[1]
    info = analyze(ticker)
    snap = live_snapshot(ticker, synthetic=info["synthetic"])
    p = float(_best[key].predict_proba(latest_features(snap["daily"]))[0, 1])
    times, prices = _thin_pair(snap["times"], snap["prices"])
    out = {
        "ticker": info["ticker"],
        "synthetic": snap["synthetic"],
        "market_open": snap["market_open"],
        "price": round(snap["price"], 4),
        "prev_close": round(snap["prev_close"], 4),
        "change": snap["price"] / snap["prev_close"] - 1,
        "p_up": p,
        "signal": "UP" if p > 0.5 else "DOWN",
        "times": times,
        "prices": prices,
        "updated": time.strftime("%H:%M:%S"),
        "source": snap["source"],
        "refresh": 5 if snap["source"] == "Angel One" else 10,
    }
    _live_cache[key] = (time.time(), out)
    return out


def candles(ticker, interval):
    key = (ticker.upper(), interval)
    hit = _candle_cache.get(key)
    if hit and time.time() - hit[0] < 4:
        return hit[1]
    info = analyze(ticker)
    rows, source = live_candles(ticker, interval, synthetic=info["synthetic"])
    out = {"interval": interval, "candles": rows, "source": source, "synthetic": info["synthetic"]}
    _candle_cache[key] = (time.time(), out)
    return out
