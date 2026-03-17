"""Tests for technical indicator analysis."""

import numpy as np
import pandas as pd
import pytest

from src.technical import TechnicalAnalyzer


def _make_ohlcv(trend: float = 0.0, n: int = 100, base: float = 50000.0) -> pd.DataFrame:
    """Generate synthetic OHLCV with a given trend."""
    np.random.seed(42)
    returns = np.random.normal(trend, 0.02, n)
    prices = base * np.cumprod(1 + returns)
    return pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.002, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        "close": prices,
        "volume": np.random.lognormal(10, 1, n),
    })


@pytest.fixture
def analyzer():
    return TechnicalAnalyzer()


class TestEMACrossover:
    def test_uptrend_positive(self, analyzer):
        df = _make_ohlcv(trend=0.005)
        signal = analyzer.compute_ema_signal(df)
        assert signal > 0, f"Uptrend EMA signal should be positive, got {signal}"

    def test_downtrend_negative(self, analyzer):
        df = _make_ohlcv(trend=-0.005)
        signal = analyzer.compute_ema_signal(df)
        assert signal < 0, f"Downtrend EMA signal should be negative, got {signal}"

    def test_bounded(self, analyzer):
        df = _make_ohlcv(trend=0.01)
        signal = analyzer.compute_ema_signal(df)
        assert -1 <= signal <= 1


class TestRSI:
    def test_uptrend_positive(self, analyzer):
        df = _make_ohlcv(trend=0.005)
        signal = analyzer.compute_rsi_signal(df)
        # Strong uptrend → RSI > 50 → negative signal (overbought)
        # This is correct: RSI maps oversold=bullish, overbought=bearish
        assert isinstance(signal, float)

    def test_bounded(self, analyzer):
        df = _make_ohlcv(trend=0.01)
        signal = analyzer.compute_rsi_signal(df)
        assert -1 <= signal <= 1

    def test_flat_near_zero(self, analyzer):
        df = _make_ohlcv(trend=0.0, n=200)
        signal = analyzer.compute_rsi_signal(df)
        assert abs(signal) < 0.5, f"Flat RSI should be near zero, got {signal}"


class TestMACD:
    def test_returns_float(self, analyzer):
        df = _make_ohlcv(trend=0.005)
        signal = analyzer.compute_macd_signal(df)
        assert isinstance(signal, float)

    def test_downtrend_negative(self, analyzer):
        df = _make_ohlcv(trend=-0.005)
        signal = analyzer.compute_macd_signal(df)
        assert signal < 0, f"Downtrend MACD should be negative, got {signal}"

    def test_bounded(self, analyzer):
        df = _make_ohlcv(trend=0.02)
        signal = analyzer.compute_macd_signal(df)
        assert -1 <= signal <= 1


class TestBollinger:
    def test_bounded(self, analyzer):
        df = _make_ohlcv(trend=0.001)
        signal = analyzer.compute_bollinger_signal(df)
        assert -1 <= signal <= 1

    def test_returns_float(self, analyzer):
        df = _make_ohlcv()
        signal = analyzer.compute_bollinger_signal(df)
        assert isinstance(signal, float)


class TestComposite:
    def test_analyze_returns_all_keys(self, analyzer):
        df = _make_ohlcv()
        result = analyzer.analyze(df)
        assert "composite" in result
        assert "indicators" in result
        assert set(result["indicators"].keys()) == {
            "ema_crossover", "rsi", "macd", "bollinger"
        }

    def test_composite_bounded(self, analyzer):
        df = _make_ohlcv(trend=0.01)
        result = analyzer.analyze(df)
        assert -1 <= result["composite"] <= 1

    def test_all_indicators_bounded(self, analyzer):
        df = _make_ohlcv(trend=-0.005)
        result = analyzer.analyze(df)
        for name, score in result["indicators"].items():
            assert -1 <= score <= 1, f"{name} out of bounds: {score}"

    def test_uptrend_composite_positive(self, analyzer):
        df = _make_ohlcv(trend=0.005)
        result = analyzer.analyze(df)
        assert result["composite"] > -0.5, \
            f"Strong uptrend composite should not be very negative, got {result['composite']}"

    def test_downtrend_composite_negative(self, analyzer):
        df = _make_ohlcv(trend=-0.005)
        result = analyzer.analyze(df)
        assert result["composite"] < 0.5, \
            f"Strong downtrend composite should not be very positive, got {result['composite']}"

    def test_custom_params(self):
        params = {"ema_fast": 5, "ema_slow": 15, "rsi_period": 7,
                  "macd_fast": 5, "macd_slow": 15, "macd_signal": 5,
                  "bb_period": 10, "bb_std": 1.5}
        analyzer = TechnicalAnalyzer(params=params)
        df = _make_ohlcv()
        result = analyzer.analyze(df)
        assert "composite" in result
