# trading-signal-engine · T3

Dual-path signal engine that fuses technical indicators with local FinBERT sentiment analysis to generate trading signals for BTC, ETH, and SOL. 84/84 tests pass without a database connection or model download.

---

## Trading Arc

| Repo | Role | Status |
|------|------|--------|
| T1 · crypto-data-pipeline | Live OHLCV ingestion · market event tagging | Shipped Mar 6 |
| T2 · trading-chart-generator | Candlestick PNGs + JSON sidecars · 25/25 tests | Shipped Mar 10 |
| **T3 · trading-signal-engine** | Technical indicators + FinBERT sentiment · 84/84 tests | Shipped Mar 16 |
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

#### Threshold-tuned continuous sentiment
The sentiment signal is computed as `probs["positive"] - probs["negative"]` over the full softmax across the three FinBERT classes, not the older argmax-with-sign approach. A confidence floor (`SENTIMENT_CONFIDENCE_THRESHOLD = 0.55`) zeroes out predictions whose top-class probability falls below the threshold, so genuinely-mixed predictions like `pos=0.41, neu=0.40, neg=0.19` collapse to neutral instead of leaking noise into the fused signal. Threshold tuned via the sweep in `notebooks/threshold_sweep.md`. The threshold is a constructor parameter, so T4 backtester sweeps can override it without code changes.

### Signal Fusion

```
signal     = 0.6 × technical_score + 0.4 × sentiment_score
confidence = 1.0 − |technical_score − sentiment_score|
```

Weights are configurable in `config.py` for T4 parameter sweeps. The winning weights from T4's sweep are written back to T3 as updated defaults — closing the feedback loop.

### Vision Demo Mode
`MockVisionAnalyzer` reads T2's JSON chart sidecars and produces deterministic chart analysis — demonstrates extensibility to real vision models without API costs.

---

## T2→T3 Integration

T2 (`trading-chart-generator`) writes a JSON sidecar next to every candlestick PNG. That sidecar carries chart-level summary stats — recent range, close, trend direction, and a volatility band — that previously nothing consumed. T3 now reads them.

```
T2 chart run                          T3 signal run
─────────────                         ─────────────
BTC_1h.png                            ChartContextReader.load_summary("BTC/USD", "1h")
BTC_1h.json  ──(charts_dir)──►          → most recent, non-stale sidecar
  ohlcv_summary:                      chart_context_score(summary) → [-1, +1]
    trend, volatility_band_pct          → TechnicalAnalyzer.analyze(df, chart_context=...)
```

`ChartContextReader` (in `src/chart_context.py`) scans the configured `CHARTS_DIR`, matches sidecars by `pair`/`timeframe`, and returns the freshest non-stale one. Missing, stale, or malformed files fall back to a neutral `0.0` — a missing chart never crashes a signal run.

`chart_context_score(summary)` maps the sidecar to a signal in `[−1, +1]`:

| Sidecar field | Contribution |
|---------------|--------------|
| `trend = "up"` | +0.5 |
| `trend = "down"` | −0.5 |
| `trend = "flat"` | 0.0 |
| `volatility_band_pct > 2%` | halves the absolute contribution (uncertain regime) |

The technical path folds this in **inside** its own share — it does not touch the outer 0.6/0.4 fusion:

```
technical_score = 0.85 × indicators + 0.15 × chart_context   (when a sidecar exists)
technical_score = 1.00 × indicators                          (no sidecar — backward compatible)
signal          = 0.6  × technical_score + 0.4 × sentiment_score
```

In live mode T3 reads real T2 sidecars from `CHARTS_DIR`. In mock mode there are no charts on disk, so `SignalEngine.synthesize_sidecar()` builds an equivalent summary straight from the synthetic OHLCV, letting the demo and tests exercise the full path without T2.

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
      "chart_context": 0.50,
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
│   ├── chart_context.py # ChartContextReader: reads T2 sidecars → context score
│   ├── vision_demo.py  # MockVisionAnalyzer: deterministic chart analysis
│   └── run.py          # CLI entry point
├── tests/              # 84 tests, all run without DB or model download
├── examples/
│   └── quick_demo.py   # Works without DB or FinBERT download
├── notebooks/
│   └── threshold_sweep.md   # FinBERT confidence-floor calibration
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
# 84/84 — all run in mock mode, no DB or model download
```

---

## Contact

Mike Ichikawa · [projects.ichikawa@gmail.com](mailto:projects.ichikawa@gmail.com) · [mtichikawa.github.io](https://mtichikawa.github.io)
