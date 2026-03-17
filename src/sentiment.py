"""Sentiment analysis using FinBERT for financial headlines.

Uses the ProsusAI/finbert model via HuggingFace transformers for local inference.
No API calls — the model downloads (~400MB) on first use and is cached.
A mock mode is available for tests and demos without model download.
"""

from src.config import FINBERT_MODEL


class SentimentAnalyzer:
    """Analyze financial headline sentiment using FinBERT or mock keywords."""

    def __init__(self, model_name: str = FINBERT_MODEL, use_mock: bool = False):
        self.model_name = model_name
        self.use_mock = use_mock
        self._pipeline = None

    def _load_model(self):
        """Lazy-load the FinBERT pipeline on first real call."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
            )

    def _mock_sentiment(self, headline: str) -> dict:
        """Keyword-based mock sentiment for tests and demos."""
        h = headline.lower()

        positive_words = ["surge", "rally", "bull", "gain", "rise", "soar",
                          "breakout", "upgrade", "adoption", "record high"]
        negative_words = ["crash", "drop", "bear", "plunge", "fall", "dump",
                          "hack", "ban", "lawsuit", "fraud", "sell-off"]

        pos_count = sum(1 for w in positive_words if w in h)
        neg_count = sum(1 for w in negative_words if w in h)

        if pos_count > neg_count:
            return {"label": "positive", "score": 0.85, "signal": 0.85}
        elif neg_count > pos_count:
            return {"label": "negative", "score": 0.80, "signal": -0.80}
        else:
            return {"label": "neutral", "score": 0.60, "signal": 0.0}

    def analyze_headline(self, headline: str) -> dict:
        """Analyze a single headline.

        Returns:
            dict with "label" (str), "score" (float), "signal" (float in [-1, 1]).
        """
        if self.use_mock:
            return self._mock_sentiment(headline)

        self._load_model()
        result = self._pipeline(headline, truncation=True, max_length=512)[0]
        label = result["label"]
        score = result["score"]

        if label == "positive":
            signal = score
        elif label == "negative":
            signal = -score
        else:
            signal = 0.0

        return {"label": label, "score": score, "signal": signal}

    def analyze_headlines(self, headlines: list[dict]) -> dict:
        """Analyze multiple headlines and aggregate sentiment.

        Args:
            headlines: list of dicts with at least a "headline" key.

        Returns:
            dict with "sentiment_score" (float in [-1, 1]) and
            "headlines_used" (list of analyzed headlines).
        """
        if not headlines:
            return {"sentiment_score": 0.0, "headlines_used": []}

        analyzed = []
        signals = []

        for h in headlines:
            text = h.get("headline", h.get("text", ""))
            result = self.analyze_headline(text)
            analyzed.append({
                "headline": text,
                "sentiment": result["signal"],
                "source": h.get("source", "unknown"),
            })
            signals.append(result["signal"])

        sentiment_score = sum(signals) / len(signals)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        return {
            "sentiment_score": sentiment_score,
            "headlines_used": analyzed,
        }
