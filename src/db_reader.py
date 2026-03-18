"""Database reader for T1 crypto-data-pipeline PostgreSQL.

Reads OHLCV candles and news headlines. Same connection pattern as T2
trading-chart-generator, extended with headline queries.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_URL


class SignalDBReader:
    """Read OHLCV candles and headlines from T1 PostgreSQL."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or DB_URL
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.db_url)
        return self._engine

    def fetch_candles(self, pair: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent OHLCV candles for a pair and timeframe.

        Returns DataFrame with columns: open_time, open, high, low, close, volume.
        """
        query = text("""
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE pair = :pair AND timeframe = :timeframe
            ORDER BY open_time DESC
            LIMIT :limit
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "pair": pair, "timeframe": timeframe, "limit": limit
            })
        return df.sort_values("open_time").reset_index(drop=True)

    def fetch_headlines(self, pair_tag: str, hours_back: int = 24,
                        limit: int = 50) -> list[dict]:
        """Fetch recent news headlines for a pair tag.

        Args:
            pair_tag: e.g. "BTC", "ETH", "SOL", "crypto", "macro".
            hours_back: how far back to look.
            limit: max headlines to return.

        Returns:
            list of dicts with "headline", "source", "published_at".
        """
        query = text("""
            SELECT headline, source, published_at
            FROM news_headlines
            WHERE pair_tag = :pair_tag
              AND published_at >= NOW() - INTERVAL ':hours hours'
            ORDER BY published_at DESC
            LIMIT :limit
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "pair_tag": pair_tag, "hours": hours_back, "limit": limit
            })
        return df.to_dict("records")
