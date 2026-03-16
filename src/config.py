"""Configuration for the trading signal engine."""

import os

# Trading pairs and timeframes
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]
TIMEFRAMES = ["1h", "4h"]

# Signal fusion weights (must sum to 1.0)
FUSION_WEIGHTS = {
    "technical": 0.6,
    "sentiment": 0.4,
}

# Technical indicator parameters
INDICATOR_PARAMS = {
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
}

# Sentiment analyzer
FINBERT_MODEL = "ProsusAI/finbert"
HEADLINES_HOURS_BACK = 24
HEADLINES_LIMIT = 50

# Signal output
SIGNAL_OUTPUT_DIR = "signals"

# Database connection (same env vars as T1/T2)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "crypto_pipeline")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
