"""Tests for chart-context fusion within the technical path."""

import numpy as np
import pandas as pd
import pytest

from src.sentiment import SentimentAnalyzer
from src.signal_engine import SignalEngine
from src.technical import TechnicalAnalyzer


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
    return [{"headline": "Market remains stable", "source": "reuters"}]


class TestTechnicalWeighting:
    def test_no_context_is_pure_indicators(self):
        df = _make_ohlcv(trend=0.003)
        analyzer = TechnicalAnalyzer()
        plain = analyzer.analyze(df)
        with_none = analyzer.analyze(df, chart_context=None)
        assert plain["composite"] == with_none["composite"]

    def test_context_shifts_composite(self):
        df = _make_ohlcv(trend=0.003)
        analyzer = TechnicalAnalyzer()
        base = analyzer.analyze(df)["composite"]
        indicator_score = float(np.mean(list(
            analyzer.analyze(df)["indicators"].values()
        )))
        boosted = analyzer.analyze(df, chart_context=1.0)["composite"]
        expected = float(np.clip(0.85 * indicator_score + 0.15 * 1.0, -1, 1))
        assert abs(boosted - expected) < 1e-9
        # A positive chart context should not pull the score below the base.
        assert boosted >= base

    def test_context_bounded(self):
        df = _make_ohlcv(trend=0.01)
        analyzer = TechnicalAnalyzer()
        for ctx in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = analyzer.analyze(df, chart_context=ctx)
            assert -1 <= result["composite"] <= 1


class TestFusionWithContext:
    def test_fusion_math_with_context(self, engine, headlines):
        df = _make_ohlcv(trend=0.003)
        signal = engine.generate_signal(
            df, headlines, "BTC/USD", "1h", chart_context=0.5,
        )
        # Outer 0.6/0.4 fusion is unchanged; technical_score already folded in context.
        expected = 0.6 * signal["technical_score"] + 0.4 * signal["sentiment_score"]
        assert abs(signal["signal"] - expected) < 0.01
        assert signal["chart_context"] == 0.5

    def test_context_changes_technical_score(self, engine, headlines):
        df = _make_ohlcv(trend=0.003)
        without = engine.generate_signal(df, headlines, "BTC/USD", "1h")
        with_ctx = engine.generate_signal(
            df, headlines, "BTC/USD", "1h", chart_context=1.0,
        )
        assert without["chart_context"] is None
        assert with_ctx["technical_score"] != without["technical_score"]

    def test_synthesize_sidecar_uptrend(self):
        df = _make_ohlcv(trend=0.01)
        sidecar = SignalEngine.synthesize_sidecar(df, "BTC/USD", "1h")
        assert sidecar["pair"] == "BTC/USD"
        assert sidecar["ohlcv_summary"]["trend"] == "up"
        assert "volatility_band_pct" in sidecar["ohlcv_summary"]
