"""Quick demo — works without DB connection or FinBERT model download.

Generates synthetic OHLCV data and uses mock sentiment to demonstrate
the full signal engine pipeline.
"""

import sys
import os
import json
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.technical import TechnicalAnalyzer
from src.sentiment import SentimentAnalyzer
from src.vision_demo import MockVisionAnalyzer
from src.signal_engine import SignalEngine


def generate_synthetic_ohlcv(n: int = 100, trend: float = 0.001,
                              base_price: float = 50000.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a configurable trend.

    Args:
        n: number of candles.
        trend: drift per candle (positive = uptrend).
        base_price: starting price.

    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    np.random.seed(42)
    returns = np.random.normal(trend, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    opens = prices * (1 + np.random.normal(0, 0.003, n))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.lognormal(10, 1, n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })


def sample_headlines() -> list[dict]:
    """Return a mix of bullish, bearish, and neutral headlines."""
    return [
        {"headline": "Bitcoin surges past $50k as institutional adoption grows",
         "source": "coindesk"},
        {"headline": "Crypto market faces regulatory uncertainty",
         "source": "reuters"},
        {"headline": "Ethereum rally continues with DeFi momentum",
         "source": "cointelegraph"},
        {"headline": "Federal Reserve holds rates steady",
         "source": "bloomberg"},
        {"headline": "Major exchange reports record trading volume",
         "source": "coindesk"},
    ]


def main():
    print("=" * 60)
    print("Trading Signal Engine — Quick Demo")
    print("=" * 60)

    # Initialize components in mock mode
    technical = TechnicalAnalyzer()
    sentiment = SentimentAnalyzer(use_mock=True)
    vision = MockVisionAnalyzer()
    engine = SignalEngine(
        technical=technical,
        sentiment=sentiment,
        vision=vision,
    )

    # Generate signals for each scenario
    scenarios = [
        ("BTC/USD", "1h", 0.002, "Uptrend"),
        ("ETH/USD", "4h", -0.003, "Downtrend"),
        ("SOL/USD", "1h", 0.0, "Sideways"),
    ]

    signals = []
    for pair, timeframe, trend, label in scenarios:
        print(f"\n--- {pair} ({timeframe}) — {label} ---")
        df = generate_synthetic_ohlcv(trend=trend)
        headlines = sample_headlines()

        # Create a mock sidecar for vision demo
        sidecar = {"price_change_pct": trend * 100 * 100}
        sidecar_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(sidecar, sidecar_file)
        sidecar_file.close()

        signal = engine.generate_signal(
            ohlcv_df=df,
            headlines=headlines,
            pair=pair,
            timeframe=timeframe,
            chart_path=f"charts/{pair.replace('/', '_')}_{timeframe}.png",
            sidecar_path=sidecar_file.name,
        )
        signals.append(signal)

        os.unlink(sidecar_file.name)

        print(f"  Technical score: {signal['technical_score']:+.4f}")
        print(f"  Sentiment score: {signal['sentiment_score']:+.4f}")
        print(f"  Fused signal:    {signal['signal']:+.4f}")
        print(f"  Confidence:      {signal['confidence']:.4f}")

        if signal.get("vision_demo"):
            vd = signal["vision_demo"]
            print(f"  Vision demo:     bull={vd['bull_score']:.2f} "
                  f"bear={vd['bear_score']:.2f}")
            print(f"                   {vd['analysis']}")

    # Save signals
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "signals"
    )
    filepath = engine.save_signals(signals, output_dir)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(signals)} signals to {filepath}")

    # Summary table
    print(f"\n{'Pair':<10} {'TF':<5} {'Signal':>8} {'Conf':>6} {'Tech':>6} {'Sent':>6}")
    print("-" * 48)
    for s in signals:
        direction = "BULL" if s["signal"] > 0.1 else "BEAR" if s["signal"] < -0.1 else "FLAT"
        print(f"{s['pair']:<10} {s['timeframe']:<5} {s['signal']:>+7.3f} "
              f"{s['confidence']:>5.2f} {s['technical_score']:>+5.2f} "
              f"{s['sentiment_score']:>+5.2f}  {direction}")


if __name__ == "__main__":
    main()
