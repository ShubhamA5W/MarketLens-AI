# Stock Market Direction Predictor (Nifty 50 / NSE stocks)

Machine learning pipeline that predicts whether a stock or index will close **higher or lower tomorrow**, evaluated with time-series-safe validation and a backtest that includes transaction costs.

**Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, yfinance

## Pipeline
1. `data.py` – downloads daily prices from Yahoo Finance (falls back to synthetic data offline)
2. `features.py` – technical indicators: returns, SMA ratios, RSI, MACD, volatility, volume change (past data only, no look-ahead)
3. `train.py` – chronological 80/20 split, walk-forward cross-validation, compares Logistic Regression / Random Forest / Gradient Boosting against a majority-class baseline, then backtests vs buy-and-hold
4. `app.py` + `pipeline.py` + `templates/index.html` – Flask web dashboard (black UI with red, green and blue accents)
5. `predict.py` – prints the model's signal for the latest trading day

## Run the website
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000, type any NSE ticker (e.g. `TCS.NS`) and click Analyze. After the models train, the page polls `/api/live` every 10 seconds: the price, intraday chart, and next-session signal update automatically and each change is added to a live signal log. Accuracy, the backtest and feature importance come from the trained models.

Live prices come from Yahoo Finance polling, not a tick stream, and NSE data can be delayed up to about 15 minutes. Outside market hours the page shows the last session and says so.

## Run from the command line
```bash
pip install -r requirements.txt
python train.py              # Nifty 50 by default
python train.py RELIANCE.NS  # or TCS.NS, INFY.NS, HDFCBANK.NS ...
python predict.py
```
Outputs: `results.png` (backtest curve + feature importance) and `model.joblib`.

## Methodology notes
- **No shuffling.** Train data always precedes test data, avoiding the look-ahead leakage that inflates many student stock projects.
- **Baseline comparison.** A model must beat "always predict the majority class" to mean anything.
- **Costs included.** Backtest charges 0.1% per position change.
- **Honest expectation.** Daily direction is close to random; accuracy of 50–56% is normal. The point is a rigorous process, not a money machine.

## Ideas to extend
- Add an LSTM and compare it with the tree models
- Add news-sentiment features
- Predict volatility instead of direction (usually more predictable)
- Build a Streamlit dashboard
- Add multi-stock portfolio backtesting

*Educational project only. Not financial advice.*

## Candlestick chart controls
- **Chart type** (with icons): Candles, Hollow, Heikin Ashi, Bars (OHLC), Line, Area
- **Candle timeframe:** 1m, 5m, 15m, 1H, 1D
- **Refresh meter:** slider for 5s, 10s, 30s or 60s updates, a progress bar showing time to the next update, and a Pause/Resume button
- Hover over the chart to see Open, High, Low and Close for any candle

## Optional: faster live prices with Angel One SmartAPI
Without setup the site uses Yahoo Finance (delayed up to about 15 minutes for NSE). With a free Angel One account the live feed refreshes every 5 seconds from SmartAPI.

1. Open an Angel One account, then create a "Market Feeds" app at smartapi.angelone.in to get an API key.
2. Enable TOTP at smartapi.angelone.in/enable-totp and copy the secret shown under the QR code.
3. `pip install -r requirements-angel.txt`
4. Copy `.env.example` to `.env` and fill in the four values.
5. Run `python app.py`. The badge shows `LIVE · Angel One`; if login fails it falls back to Yahoo Finance and says so.

Supported tickers: ^NSEI, ^BSESN, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS (see `TOKENS` in `angel.py`; add more with their symbol tokens from Angel One's instrument list).
Never commit `.env` to GitHub. The included `.gitignore` already excludes it.
