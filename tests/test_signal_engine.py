"""Tests for the signal engine fusion and output."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.sentiment import SentimentAnalyzer
from src.signal_engine import SignalEngine
from src.technical import TechnicalAnalyzer
from src.vision_demo import MockVisionAnalyzer


def _make_ohlcv(trend: float = 0.0, n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    returns = np.random.normal(trend, 0.02, n)
    prices = 50000 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.002, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices,
        "volume": np.random.lognormal(10, 1, n),
    })


@pytest.fixture
def engine():
    return SignalEngine(
        technical=TechnicalAnalyzer(),
        sentiment=SentimentAnalyzer(use_mock=True),
    )


@pytest.fixture
def headlines():
    return [
        {"headline": "Bitcoin surges past $50k", "source": "coindesk"},
        {"headline": "Market remains stable", "source": "reuters"},
    ]


class TestFusionMath:
    def test_default_weights(self, engine, headlines):
        df = _make_ohlcv(trend=0.003)
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        # Verify fusion: 0.6 * tech + 0.4 * sent
        expected = 0.6 * signal["technical_score"] + 0.4 * signal["sentiment_score"]
        assert abs(signal["signal"] - expected) < 0.01

    def test_custom_weights(self, headlines):
        engine = SignalEngine(
            technical=TechnicalAnalyzer(),
            sentiment=SentimentAnalyzer(use_mock=True),
            fusion_weights={"technical": 0.3, "sentiment": 0.7},
        )
        df = _make_ohlcv(trend=0.003)
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        expected = 0.3 * signal["technical_score"] + 0.7 * signal["sentiment_score"]
        assert abs(signal["signal"] - expected) < 0.01

    def test_signal_bounded(self, engine, headlines):
        for trend in [-0.01, -0.005, 0, 0.005, 0.01]:
            df = _make_ohlcv(trend=trend)
            signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
            assert -1 <= signal["signal"] <= 1

    def test_confidence_agreement(self, engine):
        """When tech and sentiment agree, confidence should be high."""
        df = _make_ohlcv(trend=0.005)
        positive_headlines = [
            {"headline": "Bitcoin surges and soars to record high", "source": "a"},
            {"headline": "Massive rally in crypto markets", "source": "b"},
        ]
        signal = engine.generate_signal(df, positive_headlines, "BTC/USD", "1h")
        # Both should be positive → high confidence
        if signal["technical_score"] > 0 and signal["sentiment_score"] > 0:
            assert signal["confidence"] > 0.3

    def test_confidence_bounded(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        assert 0 <= signal["confidence"] <= 1


class TestOutputSchema:
    def test_required_keys(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        required = {
            "pair", "timeframe", "timestamp", "signal", "confidence",
            "technical_score", "sentiment_score", "indicators",
            "headlines_used", "chart_path", "vision_demo",
        }
        assert required.issubset(signal.keys())

    def test_indicators_present(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        assert set(signal["indicators"].keys()) == {
            "ema_crossover", "rsi", "macd", "bollinger"
        }

    def test_pair_and_timeframe(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "ETH/USD", "4h")
        assert signal["pair"] == "ETH/USD"
        assert signal["timeframe"] == "4h"

    def test_no_vision_by_default(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        assert signal["vision_demo"] is None

    def test_vision_included_when_enabled(self, headlines):
        engine = SignalEngine(
            technical=TechnicalAnalyzer(),
            sentiment=SentimentAnalyzer(use_mock=True),
            vision=MockVisionAnalyzer(),
        )
        df = _make_ohlcv()
        signal = engine.generate_signal(
            df, headlines, "BTC/USD", "1h",
            chart_path="test.png",
        )
        assert signal["vision_demo"] is not None
        assert "bull_score" in signal["vision_demo"]


class TestSaveSignals:
    def test_save_creates_file(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = SignalEngine.save_signals([signal], tmpdir)
            assert os.path.exists(filepath)

    def test_saved_json_valid(self, engine, headlines):
        df = _make_ohlcv()
        signal = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = SignalEngine.save_signals([signal], tmpdir)
            with open(filepath) as f:
                data = json.load(f)
            assert "generated_at" in data
            assert "signals" in data
            assert len(data["signals"]) == 1

    def test_empty_signals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = SignalEngine.save_signals([], tmpdir)
            with open(filepath) as f:
                data = json.load(f)
            assert data["signals"] == []
