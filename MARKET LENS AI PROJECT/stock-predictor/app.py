try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, jsonify, render_template, request

from pipeline import analyze, candles, live

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker", "^NSEI").strip() or "^NSEI"
    try:
        return jsonify(analyze(ticker))
    except Exception as e:
        return jsonify({"error": f"Could not analyze {ticker}: {e}"}), 400


@app.get("/api/live")
def api_live():
    ticker = request.args.get("ticker", "^NSEI").strip() or "^NSEI"
    try:
        return jsonify(live(ticker))
    except Exception as e:
        return jsonify({"error": f"Live feed unavailable for {ticker}: {e}"}), 503


@app.get("/api/candles")
def api_candles():
    ticker = request.args.get("ticker", "^NSEI").strip() or "^NSEI"
    interval = request.args.get("interval", "5m")
    try:
        return jsonify(candles(ticker, interval))
    except Exception as e:
        return jsonify({"error": f"Candles unavailable for {ticker}: {e}"}), 503


if __name__ == "__main__":
    app.run(debug=True)
