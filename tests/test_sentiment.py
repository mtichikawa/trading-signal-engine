"""Tests for sentiment analysis (mock mode only — no model download)."""



import pytest



from src.sentiment import SentimentAnalyzer





@pytest.fixture

def analyzer():

    return SentimentAnalyzer(use_mock=True)





class TestMockSentiment:

    def test_positive_headline(self, analyzer):

        result = analyzer.analyze_headline("Bitcoin surges to new record high")

        assert result["label"] == "positive"

        assert result["signal"] > 0



    def test_negative_headline(self, analyzer):

        result = analyzer.analyze_headline("Crypto market crash wipes billions")

        assert result["label"] == "negative"

        assert result["signal"] < 0



    def test_neutral_headline(self, analyzer):

        result = analyzer.analyze_headline("Federal Reserve holds interest rates steady")

        assert result["label"] == "neutral"

        assert result["signal"] == 0.0



    def test_signal_bounded(self, analyzer):

        for headline in [

            "Rally continues as adoption grows",

            "Major crash and dump in markets",

            "Trading volume stable",

        ]:

            result = analyzer.analyze_headline(headline)

            assert -1 <= result["signal"] <= 1



    def test_output_keys(self, analyzer):

        result = analyzer.analyze_headline("Test headline")

        assert "label" in result

        assert "score" in result

        assert "signal" in result



    def test_probs_dict_shape(self, analyzer):

        result = analyzer.analyze_headline("Bitcoin surges to new record high")

        assert "probs" in result

        assert set(result["probs"].keys()) == {"positive", "neutral", "negative"}

        assert abs(sum(result["probs"].values()) - 1.0) < 1e-6



    def test_low_confidence_collapses_to_neutral(self):

        analyzer = SentimentAnalyzer(use_mock=True, confidence_threshold=0.99)

        result = analyzer.analyze_headline("Bitcoin surges to new record high")

        assert result["signal"] == 0.0



    def test_confidence_threshold_configurable(self):

        relaxed = SentimentAnalyzer(use_mock=True, confidence_threshold=0.10)

        strict = SentimentAnalyzer(use_mock=True, confidence_threshold=0.99)

        relaxed_signal = relaxed.analyze_headline("Crypto market crash wipes billions")["signal"]

        strict_signal = strict.analyze_headline("Crypto market crash wipes billions")["signal"]

        assert relaxed_signal < 0

        assert strict_signal == 0.0





class TestHeadlineAggregation:

    def test_mixed_headlines(self, analyzer):

        headlines = [

            {"headline": "Bitcoin surges past resistance", "source": "coindesk"},

            {"headline": "Market crash fears grow", "source": "reuters"},

            {"headline": "Trading volume remains stable", "source": "bloomberg"},

        ]

        result = analyzer.analyze_headlines(headlines)

        assert "sentiment_score" in result

        assert "headlines_used" in result

        assert len(result["headlines_used"]) == 3



    def test_all_positive(self, analyzer):

        headlines = [

            {"headline": "Bitcoin rally surges", "source": "a"},

            {"headline": "ETH gains and soars", "source": "b"},

        ]

        result = analyzer.analyze_headlines(headlines)

        assert result["sentiment_score"] > 0



    def test_all_negative(self, analyzer):

        headlines = [

            {"headline": "Massive crash and dump", "source": "a"},

            {"headline": "Market plunge continues", "source": "b"},

        ]

        result = analyzer.analyze_headlines(headlines)

        assert result["sentiment_score"] < 0



    def test_empty_headlines(self, analyzer):

        result = analyzer.analyze_headlines([])

        assert result["sentiment_score"] == 0.0

        assert result["headlines_used"] == []



    def test_sentiment_score_bounded(self, analyzer):

        headlines = [{"headline": f"Headline {i}", "source": "test"} for i in range(10)]

        result = analyzer.analyze_headlines(headlines)

        assert -1 <= result["sentiment_score"] <= 1



    def test_headline_source_preserved(self, analyzer):

        headlines = [{"headline": "Bitcoin surges", "source": "coindesk"}]

        result = analyzer.analyze_headlines(headlines)

        assert result["headlines_used"][0]["source"] == "coindesk"

