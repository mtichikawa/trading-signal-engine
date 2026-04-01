# trading-signal-engine · T3

Dual-path signal engine that fuses technical indicators with local FinBERT sentiment analysis to generate trading signals for BTC, ETH, and SOL. 51/51 tests pass without a database connection or model download.

---

## Trading Arc

| Repo | Role | Status |
|------|------|--------|
| T1 · crypto-data-pipeline | Live OHLCV ingestion · market event tagging | Shipped Mar 6 |
| T2 · trading-chart-generator | Candlestick PNGs + JSON sidecars · 25/25 tests | Shipped Mar 10 |
| **T3 · trading-signal-engine** | Technical indicators + FinBERT sentiment · 51/51 tests | Shipped Mar 16 |
| T4 · trading-backtester | Backtesting + parameter sweep · 72/72 tests | Shipped Mar 26 |
| T5 · trading-dashboard | Streamlit oversight UI · 8/8 tests | Shipped Mar 31 · [Live Demo](https://mtichikawa-trading.streamlit.app) |

---

## Architecture

### Technical Path
Pure pandas/numpy indicators computed from T1 OHLCV data:

| Indicator | Logic |
|-----------|-------|
| EMA Crossover | EMA(12) vs EMA(26), scaled by spread |
| RSI(14) | Mapped to [−1, +1]: negative = oversold (bullish) |
| MACD(12,26,9) | Histogram sign and magnitude |
| Bollinger Bands(20,2) | Price position within bands |

### Sentiment Path
Local HuggingFace model (`ProsusAI/finbert`) runs inference on news headlines from T1's `news_headlines` table. Downloads ~400MB on first use, then cached. No API calls.

### Signal Fusion

```
signal     = 0.6 × technical_score + 0.4 × sentiment_score
confidence = 1.0 − |technical_score − sentiment_score|
```

Weights are configurable in `config.py` for T4 parameter sweeps. The winning weights from T4's sweep are written back to T3 as updated defaults — closing the feedback loop.

### Vision Demo Mode
`MockVisionAnalyzer` reads T2's JSON chart sidecars and produces deterministic chart analysis — demonstrates extensibility to real vision models without API costs.

---

## Output

JSON signal files in `signals/` consumed by T4:

```json
{
  "generated_at": "2026-03-16T14:21:32+00:00",
  "signals": [
    {
      "pair": "BTC/USD",
      "timeframe": "1h",
      "signal": 0.42,
      "confidence": 0.78,
      "technical_score": 0.55,
      "sentiment_score": 0.22,
      "indicators": {
        "ema_crossover": 0.60,
        "rsi": 0.30,
        "macd": 0.70,
        "bollinger": 0.60
      }
    }
  ]
}
```

---

## Project Structure

```
trading-signal-engine/
├── src/
│   ├── config.py       # FUSION_WEIGHTS, PAIRS, TIMEFRAMES, SIGNAL_OUTPUT_DIR
│   ├── technical.py    # TechnicalAnalyzer: EMA, RSI, MACD, Bollinger (pure pandas)
│   ├── sentiment.py    # SentimentAnalyzer: local FinBERT + mock mode
│   ├── signal_engine.py # SignalEngine: orchestrates paths, writes JSON output
│   ├── db_reader.py    # Reads T1 PostgreSQL + T2 chart paths
│   ├── vision_demo.py  # MockVisionAnalyzer: deterministic chart analysis
│   └── run.py          # CLI entry point
├── tests/              # 51 tests, all run without DB or model download
├── examples/
│   └── quick_demo.py   # Works without DB or FinBERT download
├── signals/            # JSON signal output (consumed by T4)
└── requirements.txt
```

---

## Setup

```bash
cd trading-signal-engine
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Demo — no DB or model download needed
python examples/quick_demo.py

# Mock mode via CLI
python -m src.run --mock --vision-demo

# Live mode (requires T1 PostgreSQL, downloads FinBERT ~400MB on first run)
python -m src.run
```

## Tests

```bash
pytest tests/ -v
# 51/51 — all run in mock mode, no DB or model download
```

---

## Contact

Mike Ichikawa · [projects.ichikawa@gmail.com](mailto:projects.ichikawa@gmail.com) · [mtichikawa.github.io](https://mtichikawa.github.io)
