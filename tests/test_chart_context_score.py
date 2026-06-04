"""Tests for chart_context_score: trend mapping + volatility damping."""

import pytest

from src.chart_context import chart_context_score


class TestTrendMapping:
    def test_up_is_positive(self):
        assert chart_context_score({"trend": "up", "volatility_band_pct": 1.0}) == 0.5

    def test_down_is_negative(self):
        assert chart_context_score({"trend": "down", "volatility_band_pct": 1.0}) == -0.5

    def test_flat_is_zero(self):
        assert chart_context_score({"trend": "flat", "volatility_band_pct": 1.0}) == 0.0

    def test_bounded(self):
        for trend in ["up", "down", "flat"]:
            score = chart_context_score({"trend": trend, "volatility_band_pct": 0.5})
            assert -1 <= score <= 1


class TestVolatilityDamping:
    def test_high_volatility_halves_up(self):
        # band > 2% halves the absolute contribution.
        assert chart_context_score({"trend": "up", "volatility_band_pct": 3.5}) == 0.25

    def test_high_volatility_halves_down(self):
        assert chart_context_score({"trend": "down", "volatility_band_pct": 5.0}) == -0.25

    def test_at_threshold_not_damped(self):
        # Exactly 2% is not "> 2%", so full strength.
        assert chart_context_score({"trend": "up", "volatility_band_pct": 2.0}) == 0.5

    def test_low_volatility_full_strength(self):
        assert chart_context_score({"trend": "down", "volatility_band_pct": 0.8}) == -0.5


class TestMalformed:
    def test_empty_dict_neutral(self):
        assert chart_context_score({}) == 0.0

    def test_none_neutral(self):
        assert chart_context_score(None) == 0.0

    def test_unknown_trend_neutral(self):
        assert chart_context_score({"trend": "sideways"}) == 0.0

    def test_missing_volatility_uses_full_trend(self):
        # No band field → no damping.
        assert chart_context_score({"trend": "up"}) == 0.5

    def test_unparseable_volatility_uses_full_trend(self):
        assert chart_context_score({"trend": "up", "volatility_band_pct": "n/a"}) == 0.5
