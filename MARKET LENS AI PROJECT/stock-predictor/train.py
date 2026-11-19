import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_prices
from features import FEATURES, build_features

TICKER = sys.argv[1] if len(sys.argv) > 1 else "^NSEI"
COST = 0.001

df, name = load_prices(TICKER)
feat = build_features(df)
X, y = feat[FEATURES], feat["target"]

split = int(len(feat) * 0.8)
X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
print(f"Train: {X_tr.index[0].date()} -> {X_tr.index[-1].date()} ({len(X_tr)} days)")
print(f"Test : {X_te.index[0].date()} -> {X_te.index[-1].date()} ({len(X_te)} days)\n")

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
}

base_acc = max(y_te.mean(), 1 - y_te.mean())
print(f"{'Baseline (majority class)':28s} Test acc={base_acc:.3f}")

results, probs = {}, {}
for n, m in models.items():
    cv = [accuracy_score(y_tr.iloc[v], m.fit(X_tr.iloc[t], y_tr.iloc[t]).predict(X_tr.iloc[v]))
          for t, v in TimeSeriesSplit(5).split(X_tr)]
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    probs[n] = p
    results[n] = {"cv_acc": np.mean(cv), "acc": accuracy_score(y_te, p > 0.5), "auc": roc_auc_score(y_te, p)}
    r = results[n]
    print(f"{n:28s} WalkFwd CV acc={r['cv_acc']:.3f}  Test acc={r['acc']:.3f}  AUC={r['auc']:.3f}")

best = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest by AUC: {best}")

def backtest(p):
    pos = pd.Series((p > 0.5).astype(int), index=X_te.index)
    gross = pos * feat.loc[X_te.index, "next_ret"]
    cost = pos.diff().abs().fillna(0) * COST
    return (1 + (gross - cost)).cumprod()

bh = (1 + feat.loc[X_te.index, "next_ret"]).cumprod()
strat = backtest(probs[best])

def stats(curve):
    r = curve.pct_change().dropna()
    yrs = len(curve) / 252
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    dd = (curve / curve.cummax() - 1).min()
    return cagr, sharpe, dd

print(f"\n{'Backtest (test period)':28s} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
for label, curve in (("Model strategy", strat), ("Buy & hold", bh)):
    c, s, d = stats(curve)
    print(f"{label:28s} {c:8.1%} {s:8.2f} {d:8.1%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(strat, label=f"{best} strategy"); ax[0].plot(bh, label="Buy & hold")
ax[0].set_title(f"Backtest on unseen data ({name})"); ax[0].legend(); ax[0].grid(alpha=.3)
mdl = models["RandomForest"]
fi = pd.Series(mdl.feature_importances_, index=FEATURES).sort_values()
fi.plot.barh(ax=ax[1], title="Feature importance (Random Forest)")
plt.tight_layout(); plt.savefig("results.png", dpi=130)

joblib.dump({"model": models[best], "name": best, "ticker": name}, "model.joblib")
print("\nSaved results.png and model.joblib")
