"""Sentiment analysis using FinBERT for financial headlines.

Uses the ProsusAI/finbert model via HuggingFace transformers for local inference.
No API calls — the model downloads (~400MB) on first use and is cached.
A mock mode is available for tests and demos without model download.

The signal is computed as ``probs["positive"] - probs["negative"]`` over the
full softmax across the three FinBERT classes. A confidence floor zeroes out
predictions whose top-class probability falls below ``confidence_threshold``,
so genuinely-mixed headlines collapse to neutral instead of leaking noise
into the fused signal downstream.
"""

from src.config import FINBERT_MODEL, SENTIMENT_CONFIDENCE_THRESHOLD


class SentimentAnalyzer:
    """Analyze financial headline sentiment using FinBERT or mock keywords."""

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        use_mock: bool = False,
        confidence_threshold: float = SENTIMENT_CONFIDENCE_THRESHOLD,
    ):
        self.model_name = model_name
        self.use_mock = use_mock
        self.confidence_threshold = confidence_threshold
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
            probs = {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
        elif neg_count > pos_count:
            probs = {"positive": 0.05, "neutral": 0.15, "negative": 0.80}
        else:
            probs = {"positive": 0.20, "neutral": 0.60, "negative": 0.20}

        return self._build_result(probs)

    def _build_result(self, probs: dict) -> dict:
        """Apply confidence floor and shape the output dict."""
        label = max(probs, key=probs.get)
        max_prob = probs[label]

        if max_prob < self.confidence_threshold:
            signal = 0.0
        else:
            signal = probs.get("positive", 0.0) - probs.get("negative", 0.0)

        return {
            "label": label,
            "score": max_prob,
            "signal": signal,
            "probs": probs,
        }

    def analyze_headline(self, headline: str) -> dict:
        """Analyze a single headline.

        Returns:
            dict with ``label`` (str), ``score`` (float, top-class probability),
            ``signal`` (float in [-1, 1], pos minus neg, floored to 0 below
            ``confidence_threshold``), and ``probs`` (dict of all three class
            probabilities).
        """
        if self.use_mock:
            return self._mock_sentiment(headline)

        self._load_model()
        result = self._pipeline(headline, truncation=True, max_length=512, top_k=None)
        # top_k=None returns either a flat list of dicts or a list-of-lists
        # depending on transformers version; normalise.
        if result and isinstance(result[0], list):
            result = result[0]
        probs = {r["label"]: r["score"] for r in result}

        return self._build_result(probs)

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
