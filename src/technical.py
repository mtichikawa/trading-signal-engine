"""Technical indicator analysis on OHLCV data.

All indicators are computed with pure pandas/numpy — no external TA libraries.
Each indicator produces a signal in [-1, +1]. The composite score is the
equally-weighted average of all indicator signals.
"""

import numpy as np
import pandas as pd

from src.config import INDICATOR_PARAMS


class TechnicalAnalyzer:
    """Compute technical indicator signals from OHLCV DataFrames."""

    def __init__(self, params: dict | None = None):
        self.params = params or INDICATOR_PARAMS

    def compute_ema_signal(self, df: pd.DataFrame) -> float:
        """EMA crossover signal: fast EMA vs slow EMA.

        Returns +1 when fast is far above slow (strong uptrend),
        -1 when fast is far below slow (strong downtrend).
        """
        close = df["close"]
        fast = close.ewm(span=self.params["ema_fast"], adjust=False).mean()
        slow = close.ewm(span=self.params["ema_slow"], adjust=False).mean()

        spread = (fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1]
        return float(np.clip(spread * 20, -1, 1))

    def compute_rsi_signal(self, df: pd.DataFrame) -> float:
        """RSI(14) mapped to [-1, +1].

        RSI < 30 (oversold) → positive signal (bullish).
        RSI > 70 (overbought) → negative signal (bearish).
        """
        close = df["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.params["rsi_period"]).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        if np.isnan(rsi_val):
            return 0.0

        return float(np.clip((50 - rsi_val) / 50, -1, 1))

    def compute_macd_signal(self, df: pd.DataFrame) -> float:
        """MACD histogram sign and magnitude.

        Positive histogram → bullish, negative → bearish.
        Magnitude scaled relative to price.
        """
        close = df["close"]
        fast_ema = close.ewm(span=self.params["macd_fast"], adjust=False).mean()
        slow_ema = close.ewm(span=self.params["macd_slow"], adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params["macd_signal"], adjust=False).mean()
        histogram = macd_line - signal_line

        price = close.iloc[-1]
        if price == 0:
            return 0.0

        normalized = histogram.iloc[-1] / price * 100
        return float(np.clip(normalized * 10, -1, 1))

    def compute_bollinger_signal(self, df: pd.DataFrame) -> float:
        """Position within Bollinger Bands.

        Near lower band → +1 (bullish), near upper band → -1 (bearish).
        """
        close = df["close"]
        sma = close.rolling(self.params["bb_period"]).mean()
        std = close.rolling(self.params["bb_period"]).std()

        upper = sma + self.params["bb_std"] * std
        lower = sma - self.params["bb_std"] * std

        current = close.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]

        if np.isnan(upper_val) or np.isnan(lower_val):
            return 0.0

        band_width = upper_val - lower_val
        if band_width == 0:
            return 0.0

        position = (current - lower_val) / band_width
        return float(np.clip(1 - 2 * position, -1, 1))

    def analyze(self, df: pd.DataFrame) -> dict:
        """Run all indicators and return composite score.

        Returns:
            dict with "composite" (float in [-1, 1]) and "indicators" (dict of scores).
        """
        indicators = {
            "ema_crossover": self.compute_ema_signal(df),
            "rsi": self.compute_rsi_signal(df),
            "macd": self.compute_macd_signal(df),
            "bollinger": self.compute_bollinger_signal(df),
        }

        composite = float(np.mean(list(indicators.values())))
        composite = float(np.clip(composite, -1, 1))

        return {
            "composite": composite,
            "indicators": indicators,
        }
