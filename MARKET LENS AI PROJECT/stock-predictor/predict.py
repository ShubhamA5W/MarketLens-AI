import sys
import joblib
from data import load_prices
from features import latest_features

bundle = joblib.load("model.joblib")
ticker = sys.argv[1] if len(sys.argv) > 1 else bundle["ticker"]
df, name = load_prices(ticker)

last = latest_features(df)
p = bundle["model"].predict_proba(last)[0, 1]
print(f"{name} | model: {bundle['name']} | features as of {last.index[0].date()}")
print(f"P(next day closes higher) = {p:.1%}  ->  {'UP' if p > .5 else 'DOWN'} signal")
print("Educational project only. Not financial advice.")
