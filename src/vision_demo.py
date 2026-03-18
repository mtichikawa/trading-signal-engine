"""Mock vision analyzer for chart analysis demo.

Follows the llm-bias-detection mock response pattern: deterministic responses
derived from chart sidecar data, no API calls required. If ANTHROPIC_API_KEY
is set and use_api=True, can optionally call Claude Vision (not default).
"""

import json
import os


class MockVisionAnalyzer:
    """Analyze chart PNGs using mock responses derived from JSON sidecars."""

    def analyze_chart(self, chart_path: str, sidecar_path: str | None = None) -> dict:
        """Generate a mock vision analysis for a chart.

        Args:
            chart_path: path to chart PNG file.
            sidecar_path: path to JSON sidecar with OHLCV summary stats.

        Returns:
            dict with analysis, bull/bear scores, patterns, and reasoning.
        """
        sidecar = self._load_sidecar(sidecar_path)
        price_change = sidecar.get("price_change_pct", 0.0)

        if price_change > 2.0:
            bull_score = min(0.9, 0.6 + price_change / 20)
            bear_score = 1.0 - bull_score
            patterns = ["bullish_engulfing", "higher_highs"]
            analysis = "Strong bullish momentum with consecutive higher highs."
            reasoning = (
                f"Price gained {price_change:.1f}% over the period. "
                "EMA crossover confirmed uptrend with increasing volume."
            )
        elif price_change < -2.0:
            bear_score = min(0.9, 0.6 + abs(price_change) / 20)
            bull_score = 1.0 - bear_score
            patterns = ["bearish_engulfing", "lower_lows"]
            analysis = "Bearish pressure with breakdown below support."
            reasoning = (
                f"Price dropped {abs(price_change):.1f}% over the period. "
                "MACD histogram turned negative, confirming downtrend."
            )
        else:
            bull_score = 0.50
            bear_score = 0.50
            patterns = ["doji", "consolidation"]
            analysis = "Consolidation pattern with no clear directional bias."
            reasoning = (
                f"Price moved {price_change:+.1f}% — within noise range. "
                "Bollinger Bands narrowing suggests breakout is pending."
            )

        return {
            "chart_path": chart_path,
            "analysis": analysis,
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
            "patterns_detected": patterns,
            "reasoning": reasoning,
        }

    def _load_sidecar(self, sidecar_path: str | None) -> dict:
        """Load JSON sidecar or return defaults."""
        if sidecar_path and os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                return json.load(f)
        return {"price_change_pct": 0.0}
