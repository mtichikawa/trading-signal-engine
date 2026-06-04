"""Tests for the T2 sidecar reader."""

from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from src.chart_context import ChartContextReader


def _write_sidecar(directory, name, pair, timeframe, trend,
                   volatility_band_pct=1.0, generated_at=None):
    """Write a T2-shaped sidecar JSON and return its path."""
    sidecar = {
        "pair": pair,
        "timeframe": timeframe,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "ohlcv_summary": {
            "high": 65432.10,
            "low": 64100.00,
            "close": 65100.00,
            "trend": trend,
            "volatility_band_pct": volatility_band_pct,
        },
        "near_event": False,
        "event_type": None,
    }
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        json.dump(sidecar, f)
    return path


class TestHappyPath:
    def test_loads_matching_sidecar(self, tmp_path):
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "up")
        reader = ChartContextReader(str(tmp_path))
        summary = reader.load_summary("BTC/USD", "1h")
        assert summary is not None
        assert summary["trend"] == "up"

    def test_score_from_sidecar(self, tmp_path):
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "down")
        reader = ChartContextReader(str(tmp_path))
        assert reader.score("BTC/USD", "1h") == -0.5

    def test_picks_most_recent(self, tmp_path):
        older = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        newer = datetime.now(timezone.utc).isoformat()
        _write_sidecar(tmp_path, "old.json", "BTC/USD", "1h", "down",
                       generated_at=older)
        _write_sidecar(tmp_path, "new.json", "BTC/USD", "1h", "up",
                       generated_at=newer)
        reader = ChartContextReader(str(tmp_path))
        summary = reader.load_summary("BTC/USD", "1h")
        assert summary["trend"] == "up"

    def test_matches_pair_and_timeframe(self, tmp_path):
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "up")
        _write_sidecar(tmp_path, "eth_4h.json", "ETH/USD", "4h", "down")
        reader = ChartContextReader(str(tmp_path))
        assert reader.load_summary("ETH/USD", "4h")["trend"] == "down"


class TestMissing:
    def test_missing_directory(self):
        reader = ChartContextReader("/nonexistent/charts/dir")
        assert reader.load_summary("BTC/USD", "1h") is None
        assert reader.score("BTC/USD", "1h") == 0.0

    def test_no_matching_sidecar(self, tmp_path):
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "up")
        reader = ChartContextReader(str(tmp_path))
        assert reader.load_summary("SOL/USD", "1h") is None
        assert reader.score("SOL/USD", "1h") == 0.0

    def test_empty_directory(self, tmp_path):
        reader = ChartContextReader(str(tmp_path))
        assert reader.load_summary("BTC/USD", "1h") is None


class TestStale:
    def test_stale_sidecar_ignored(self, tmp_path):
        stale = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        _write_sidecar(tmp_path, "old.json", "BTC/USD", "1h", "up",
                       generated_at=stale)
        reader = ChartContextReader(str(tmp_path), max_age_minutes=60)
        assert reader.load_summary("BTC/USD", "1h") is None
        assert reader.score("BTC/USD", "1h") == 0.0

    def test_fresh_within_window(self, tmp_path):
        recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "up",
                       generated_at=recent)
        reader = ChartContextReader(str(tmp_path), max_age_minutes=60)
        assert reader.load_summary("BTC/USD", "1h") is not None


class TestMalformed:
    def test_malformed_json_skipped(self, tmp_path):
        bad = os.path.join(tmp_path, "bad.json")
        with open(bad, "w") as f:
            f.write("{not valid json")
        # A valid one alongside the bad one should still be found.
        _write_sidecar(tmp_path, "btc_1h.json", "BTC/USD", "1h", "up")
        reader = ChartContextReader(str(tmp_path))
        assert reader.load_summary("BTC/USD", "1h")["trend"] == "up"

    def test_only_malformed_returns_none(self, tmp_path):
        bad = os.path.join(tmp_path, "bad.json")
        with open(bad, "w") as f:
            f.write("[1, 2, 3]")  # valid JSON, wrong shape
        reader = ChartContextReader(str(tmp_path))
        assert reader.load_summary("BTC/USD", "1h") is None
