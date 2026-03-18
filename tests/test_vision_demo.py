"""Tests for the mock vision analyzer."""

import json
import os
import tempfile

import pytest

from src.vision_demo import MockVisionAnalyzer


@pytest.fixture
def analyzer():
    return MockVisionAnalyzer()


class TestMockOutput:
    def test_output_keys(self, analyzer):
        result = analyzer.analyze_chart("test.png")
        required = {"chart_path", "analysis", "bull_score", "bear_score",
                     "patterns_detected", "reasoning"}
        assert required.issubset(result.keys())

    def test_chart_path_preserved(self, analyzer):
        result = analyzer.analyze_chart("charts/BTC_1h.png")
        assert result["chart_path"] == "charts/BTC_1h.png"

    def test_scores_sum_to_one(self, analyzer):
        result = analyzer.analyze_chart("test.png")
        assert abs(result["bull_score"] + result["bear_score"] - 1.0) < 0.01


class TestPriceDerivedScoring:
    def test_bullish_on_positive_change(self, analyzer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"price_change_pct": 5.0}, f)
            sidecar = f.name
        try:
            result = analyzer.analyze_chart("test.png", sidecar)
            assert result["bull_score"] > result["bear_score"]
            assert "bullish" in result["patterns_detected"][0]
        finally:
            os.unlink(sidecar)

    def test_bearish_on_negative_change(self, analyzer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"price_change_pct": -5.0}, f)
            sidecar = f.name
        try:
            result = analyzer.analyze_chart("test.png", sidecar)
            assert result["bear_score"] > result["bull_score"]
            assert "bearish" in result["patterns_detected"][0]
        finally:
            os.unlink(sidecar)

    def test_neutral_on_flat(self, analyzer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"price_change_pct": 0.5}, f)
            sidecar = f.name
        try:
            result = analyzer.analyze_chart("test.png", sidecar)
            assert result["bull_score"] == 0.50
            assert result["bear_score"] == 0.50
        finally:
            os.unlink(sidecar)

    def test_bull_score_bounded(self, analyzer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"price_change_pct": 100.0}, f)
            sidecar = f.name
        try:
            result = analyzer.analyze_chart("test.png", sidecar)
            assert result["bull_score"] <= 1.0
            assert result["bear_score"] >= 0.0
        finally:
            os.unlink(sidecar)


class TestMissingSidecar:
    def test_no_sidecar_returns_neutral(self, analyzer):
        result = analyzer.analyze_chart("test.png")
        assert result["bull_score"] == 0.50
        assert result["bear_score"] == 0.50

    def test_nonexistent_sidecar_path(self, analyzer):
        result = analyzer.analyze_chart("test.png", "/nonexistent/path.json")
        assert result["bull_score"] == 0.50

    def test_none_sidecar(self, analyzer):
        result = analyzer.analyze_chart("test.png", None)
        assert "analysis" in result
