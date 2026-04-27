"""Signal engine: orchestrates technical + sentiment paths and fuses scores."""

import json
import os
from datetime import datetime, timezone

import numpy as np

from src.config import FUSION_WEIGHTS, PAIRS, SIGNAL_OUTPUT_DIR, TIMEFRAMES
from src.sentiment import SentimentAnalyzer
from src.technical import TechnicalAnalyzer
from src.vision_demo import MockVisionAnalyzer


class SignalEngine:
    """Dual-path signal engine with configurable fusion weights.

    Fused signal = tech_weight * technical_score + sent_weight * sentiment_score.
    Confidence = 1.0 - |technical_score - sentiment_score|.
    """

    def __init__(
        self,
        technical: TechnicalAnalyzer | None = None,
        sentiment: SentimentAnalyzer | None = None,
        vision: MockVisionAnalyzer | None = None,
        fusion_weights: dict | None = None,
    ):
        self.technical = technical or TechnicalAnalyzer()
        self.sentiment = sentiment or SentimentAnalyzer(use_mock=True)  # live mode downloads ~400MB FinBERT
        self.vision = vision
        self.weights = fusion_weights or FUSION_WEIGHTS  # weights must sum to 1.0

    def generate_signal(
        self,
        ohlcv_df,
        headlines: list[dict],
        pair: str,
        timeframe: str,
        chart_path: str | None = None,
        sidecar_path: str | None = None,
    ) -> dict:
        """Generate a fused signal from technical indicators and sentiment.

        Args:
            ohlcv_df: DataFrame with open, high, low, close, volume columns.
            headlines: list of headline dicts for sentiment analysis.
            pair: trading pair (e.g. "BTC/USD").
            timeframe: candle timeframe (e.g. "1h").
            chart_path: optional path to chart PNG for vision demo.
            sidecar_path: optional path to chart JSON sidecar.

        Returns:
            Signal dict with scores, indicators, and metadata.
        """
        tech_result = self.technical.analyze(ohlcv_df)
        sent_result = self.sentiment.analyze_headlines(headlines)

        technical_score = tech_result["composite"]
        sentiment_score = sent_result["sentiment_score"]

        w_tech = self.weights["technical"]
        w_sent = self.weights["sentiment"]
        signal = w_tech * technical_score + w_sent * sentiment_score
        signal = float(np.clip(signal, -1, 1))

        confidence = 1.0 - abs(technical_score - sentiment_score)
        confidence = max(0.0, min(1.0, confidence))

        vision_demo = None
        if self.vision and chart_path:
            vision_demo = self.vision.analyze_chart(chart_path, sidecar_path)

        return {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": round(signal, 4),
            "confidence": round(confidence, 4),
            "technical_score": round(technical_score, 4),
            "sentiment_score": round(sentiment_score, 4),
            "indicators": tech_result["indicators"],
            "headlines_used": sent_result["headlines_used"],
            "chart_path": chart_path,
            "vision_demo": vision_demo,
        }

    def run(self, reader=None, pairs=None, timeframes=None,
            enable_vision=False) -> list[dict]:
        """Run signal generation across all pairs and timeframes.

        Args:
            reader: SignalDBReader instance. If None, raises error.
            pairs: override default pairs list.
            timeframes: override default timeframes list.
            enable_vision: if True and self.vision is set, run vision demo.

        Returns:
            list of signal dicts.
        """
        if reader is None:
            raise ValueError("reader is required for live signal generation")

        pairs = pairs or PAIRS
        timeframes = timeframes or TIMEFRAMES
        signals = []

        for pair in pairs:
            pair_tag = pair.split("/")[0]
            headlines = reader.fetch_headlines(pair_tag)

            for tf in timeframes:
                df = reader.fetch_candles(pair, tf)
                if df.empty:
                    continue

                signal = self.generate_signal(
                    ohlcv_df=df,
                    headlines=headlines,
                    pair=pair,
                    timeframe=tf,
                )
                signals.append(signal)

        return signals

    @staticmethod
    def save_signals(signals: list[dict], output_dir: str | None = None) -> str:
        """Write signals to a timestamped JSON file.

        Returns the output file path.
        """
        output_dir = output_dir or SIGNAL_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"signals_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        return filepath
