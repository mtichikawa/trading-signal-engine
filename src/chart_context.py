"""Chart-context features derived from T2 trading-chart-generator sidecars.

T2 writes a JSON sidecar alongside each candlestick PNG with OHLCV summary
stats (recent range, trend direction, volatility band). This module reads the
most recent sidecar per (pair, timeframe) and turns its trend + volatility band
into a chart-context signal in [-1, +1] that the technical path can weight.

Everything degrades gracefully: missing, stale, or malformed sidecars fall back
to a neutral 0.0 contribution rather than raising — the signal engine should
never crash because a chart file is absent.
"""

from datetime import datetime, timezone
import glob
import json
import os

# Sidecars older than this are treated as stale and ignored (the chart no
# longer reflects current price action). Generous default — T2 regenerates
# charts on its own cadence and we don't want to drop a slightly-old chart.
DEFAULT_MAX_AGE_MINUTES = 360

# Volatility band above this (percent) marks an uncertain regime; the trend
# contribution is halved because direction is less trustworthy in chop.
HIGH_VOLATILITY_BAND_PCT = 2.0


def chart_context_score(summary: dict) -> float:
    """Map a sidecar's OHLCV summary to a chart-context signal in [-1, +1].

    Scoring (per the T2→T3 sidecar contract):
        trend "up"   → +0.5
        trend "down" → -0.5
        trend "flat" →  0.0
    A high volatility band (> 2%) halves the absolute contribution, since
    direction is less reliable in a choppy regime.

    Malformed or missing summaries fall back to 0.0 (neutral).
    """
    if not isinstance(summary, dict):
        return 0.0

    trend = summary.get("trend")
    if trend == "up":
        score = 0.5
    elif trend == "down":
        score = -0.5
    else:
        # "flat", None, or anything unexpected → neutral.
        return 0.0

    band = summary.get("volatility_band_pct")
    try:
        if band is not None and float(band) > HIGH_VOLATILITY_BAND_PCT:
            score *= 0.5
    except (TypeError, ValueError):
        # Unparseable band — keep the full-strength trend contribution.
        pass

    return float(score)


class ChartContextReader:
    """Load the most recent T2 sidecar JSON per (pair, timeframe)."""

    def __init__(self, charts_dir: str, max_age_minutes: int = DEFAULT_MAX_AGE_MINUTES):
        self.charts_dir = charts_dir
        self.max_age_minutes = max_age_minutes

    def load_summary(self, pair: str, timeframe: str) -> dict | None:
        """Return the OHLCV summary from the most recent matching sidecar.

        Sidecars are matched by the ``pair``/``timeframe`` fields inside the
        JSON, and the freshest (by ``generated_at``, falling back to file mtime)
        non-stale one wins. Returns None when nothing usable is found.
        """
        if not self.charts_dir or not os.path.isdir(self.charts_dir):
            return None

        best = None
        best_ts = None
        for path in glob.glob(os.path.join(self.charts_dir, "*.json")):
            sidecar = self._load_file(path)
            if sidecar is None:
                continue
            if sidecar.get("pair") != pair or sidecar.get("timeframe") != timeframe:
                continue

            ts = self._sidecar_timestamp(sidecar, path)
            if self._is_stale(ts):
                continue

            if best_ts is None or ts > best_ts:
                best, best_ts = sidecar, ts

        if best is None:
            return None
        return best.get("ohlcv_summary")

    def score(self, pair: str, timeframe: str) -> float:
        """Convenience: load the freshest summary and score it ([-1, +1]).

        Falls back to 0.0 (neutral) when no usable sidecar exists.
        """
        summary = self.load_summary(pair, timeframe)
        if summary is None:
            return 0.0
        return chart_context_score(summary)

    def _load_file(self, path: str) -> dict | None:
        """Read and parse a sidecar; return None on any read/parse error."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _sidecar_timestamp(self, sidecar: dict, path: str) -> datetime:
        """Best-effort timestamp for a sidecar: generated_at, else file mtime."""
        generated_at = sidecar.get("generated_at")
        if isinstance(generated_at, str):
            try:
                # Tolerate a trailing "Z" (not handled by fromisoformat pre-3.11).
                ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            except ValueError:
                pass
        return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)

    def _is_stale(self, ts: datetime) -> bool:
        """True when the sidecar is older than max_age_minutes."""
        age = datetime.now(timezone.utc) - ts
        return age.total_seconds() > self.max_age_minutes * 60
