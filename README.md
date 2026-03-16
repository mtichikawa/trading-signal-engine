# Trading Signal Engine (T3)

Dual-path signal engine that fuses technical indicators with FinBERT sentiment analysis to generate trading signals for crypto pairs.

Part of the Trading System Arc: T1 (data pipeline) → T2 (chart generator) → **T3 (signal engine)** → T4 (backtester) → T5 (dashboard).

## Architecture

### Path 1: Technical Indicators
Pure pandas/numpy implementation of four indicators computed from T1 OHLCV data:

| Indicator | Signal Logic |
|---|---|
| EMA Crossover | EMA(12) vs EMA(26), scaled by spread |
| RSI(14) | Mapped to [-1, +1]: oversold = bullish |
| MACD(12,26,9) | Histogram sign and magnitude |
| Bollinger Bands(20,2) | Position within bands |

### Path 2: FinBERT Sentiment
Local HuggingFace model (`ProsusAI/finbert`) runs inference on news headlines from T1's `news_headlines` table. No API calls — downloads ~400MB on first use, then cached.

### Signal Fusion
```
signal = 0.6 × technical_score + 0.4 × sentiment_score
confidence = 1.0 - |technical - sentiment|
```
Weights are configurable in `config.py` for T4 backtester parameter sweeps.

### Vision Demo Mode
Mock chart analysis following the `llm-bias-detection` pattern — deterministic responses derived from chart sidecar data. Demonstrates extensibility to Claude Vision without requiring API credits.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Demo (no DB or model download needed)
python examples/quick_demo.py

# With T1 database connection
python -m src.run

# Mock mode via CLI
python -m src.run --mock --vision-demo
```

## Output

JSON files in `signals/` with per-pair, per-timeframe signal data:
```json
{
  "pair": "BTC/USD",
  "timeframe": "1h",
  "signal": 0.42,
  "confidence": 0.78,
  "technical_score": 0.55,
  "sentiment_score": 0.22,
  "indicators": {"ema_crossover": 0.6, "rsi": 0.3, "macd": 0.7, "bollinger": 0.6}
}
```

## Testing

```bash
pytest tests/ -v
```
All tests run without DB connection or model download (mock sentiment mode).

## Tech Stack

Python, pandas, NumPy, HuggingFace Transformers (FinBERT), SQLAlchemy, PostgreSQL
