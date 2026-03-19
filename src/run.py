"""CLI entry point for the trading signal engine."""

import argparse
import json
import sys

from src.config import PAIRS, TIMEFRAMES
from src.sentiment import SentimentAnalyzer
from src.signal_engine import SignalEngine
from src.technical import TechnicalAnalyzer
from src.vision_demo import MockVisionAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Trading Signal Engine")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data and mock sentiment (no DB/model)")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Trading pairs to analyze")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Timeframes to analyze")
    parser.add_argument("--vision-demo", action="store_true",
                        help="Enable mock vision analysis in output")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for signal JSON files")
    args = parser.parse_args()

    if args.mock:
        _run_mock(args)
    else:
        _run_live(args)


def _run_mock(args):
    """Run with synthetic data — no DB or model download required."""
    from examples.quick_demo import generate_synthetic_ohlcv, sample_headlines

    pairs = args.pairs or PAIRS
    timeframes = args.timeframes or TIMEFRAMES

    sentiment = SentimentAnalyzer(use_mock=True)
    technical = TechnicalAnalyzer()
    vision = MockVisionAnalyzer() if args.vision_demo else None
    engine = SignalEngine(technical=technical, sentiment=sentiment, vision=vision)

    signals = []
    for pair in pairs:
        for tf in timeframes:
            df = generate_synthetic_ohlcv()
            signal = engine.generate_signal(
                ohlcv_df=df,
                headlines=sample_headlines(),
                pair=pair,
                timeframe=tf,
            )
            signals.append(signal)

    filepath = engine.save_signals(signals, args.output_dir)
    print(f"Generated {len(signals)} signals → {filepath}")
    _print_summary(signals)


def _run_live(args):
    """Run with T1 database connection and FinBERT model."""
    from src.db_reader import SignalDBReader

    reader = SignalDBReader()
    sentiment = SentimentAnalyzer(use_mock=False)
    technical = TechnicalAnalyzer()
    vision = MockVisionAnalyzer() if args.vision_demo else None
    engine = SignalEngine(technical=technical, sentiment=sentiment, vision=vision)

    signals = engine.run(
        reader=reader,
        pairs=args.pairs,
        timeframes=args.timeframes,
    )

    filepath = engine.save_signals(signals, args.output_dir)
    print(f"Generated {len(signals)} signals → {filepath}")
    _print_summary(signals)


def _print_summary(signals):
    """Print a formatted summary of generated signals."""
    print(f"\n{'Pair':<10} {'TF':<5} {'Signal':>8} {'Conf':>6} {'Tech':>6} {'Sent':>6}")
    print("-" * 48)
    for s in signals:
        direction = "BULL" if s["signal"] > 0.1 else "BEAR" if s["signal"] < -0.1 else "FLAT"
        print(f"{s['pair']:<10} {s['timeframe']:<5} {s['signal']:>+7.3f} "
              f"{s['confidence']:>5.2f} {s['technical_score']:>+5.2f} "
              f"{s['sentiment_score']:>+5.2f}  {direction}")


if __name__ == "__main__":
    main()
