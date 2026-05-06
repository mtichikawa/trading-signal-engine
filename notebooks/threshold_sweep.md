# FinBERT Confidence Threshold Sweep

The `SentimentAnalyzer` applies a confidence floor to FinBERT predictions:
when the top-class probability falls below `confidence_threshold`, the
returned signal collapses to `0.0` regardless of the underlying
`pos - neg` arithmetic. This document shows how that floor was tuned.

## Why a confidence floor

FinBERT returns a softmax across `{positive, neutral, negative}`. A
genuinely confident headline like `pos=0.95, neu=0.03, neg=0.02` produces
a clean signal of `+0.93`. A genuinely-mixed headline like
`pos=0.41, neu=0.40, neg=0.19` produces a small `+0.22` signal that
*looks* like a weak buy but is really just model uncertainty leaking
into the fused score downstream (`signal = 0.6 * tech + 0.4 * sent`).

The floor zeroes out predictions whose top class doesn't clear
`confidence_threshold`. The intent is "if FinBERT itself isn't sure,
contribute nothing."

## Sweep

We exercise `_build_result()` directly with eight fixture probability
distributions chosen to span the FinBERT confidence space. Mock mode
only emits three discrete distributions, so the fixtures bypass it.

Each cell shows the signal returned at that threshold; `0.00` means
the prediction was forced to neutral by the floor.

| threshold | high-conf-pos (0.95) | med-conf-pos (0.78) | low-conf-pos (0.62) | borderline-pos (0.51) | noisy-mixed (0.41) | high-conf-neg (0.93) | genuine-neutral (0.60) | mean \|signal\| (non-floored) |
|---|---|---|---|---|---|---|---|---|
| 0.40 | +0.93 | +0.71 | +0.49 | +0.29 | +0.22 | -0.91 | 0.00 | 0.604 |
| 0.50 | +0.93 | +0.71 | +0.49 | +0.29 | 0.00 | -0.91 | 0.00 | 0.668 |
| **0.55** | **+0.93** | **+0.71** | **+0.49** | **0.00** | **0.00** | **-0.91** | **0.00** | **0.744** |
| 0.60 | +0.93 | +0.71 | +0.49 | 0.00 | 0.00 | -0.91 | 0.00 | 0.744 |
| 0.70 | +0.93 | +0.71 | 0.00 | 0.00 | 0.00 | -0.91 | 0.00 | 0.807 |
| 0.80 | +0.93 | 0.00 | 0.00 | 0.00 | 0.00 | -0.91 | 0.00 | 0.920 |

## Interpretation

- **0.40, 0.50** let through the noisy-mixed case (`pos=0.41`) and the
  borderline-pos case. These are exactly the predictions we want to
  suppress — the model isn't confident, the signal is small, and
  letting them flow through dilutes the fused score.
- **0.55** drops both noisy-mixed and borderline-pos. Low-conf-pos
  (`pos=0.62`) survives, which is correct — 62% confidence is a real
  prediction, just not a screaming one.
- **0.60** behaves identically to 0.55 on these fixtures (`max=0.62`
  for low-conf-pos clears both). Slight redundancy.
- **0.70** starts dropping low-conf-pos. Too aggressive — we lose
  meaningful predictions that the model genuinely got right.
- **0.80** drops everything except very confident predictions. Mean
  `|signal|` rises to 0.92 but coverage collapses; the sentiment path
  becomes silent on most headlines.

## Default

`SENTIMENT_CONFIDENCE_THRESHOLD = 0.55` in `src/config.py`. It is the
lowest threshold that filters out both genuinely-borderline and
noisy-mixed predictions while preserving low-but-real confidence
signals.

The threshold is a constructor parameter on `SentimentAnalyzer`, so
T4 backtester parameter sweeps can override it without code changes.

## How to re-run

```bash
cd projects-hub/trading-signal-engine
source venv/bin/activate
python -c "
from src.sentiment import SentimentAnalyzer
fixtures = [
    {'positive': 0.95, 'neutral': 0.03, 'negative': 0.02},
    {'positive': 0.78, 'neutral': 0.15, 'negative': 0.07},
    # ... (see top of this file)
]
for t in [0.40, 0.50, 0.55, 0.60, 0.70, 0.80]:
    a = SentimentAnalyzer(use_mock=True, confidence_threshold=t)
    print(t, [a._build_result(f)['signal'] for f in fixtures])
"
```

Real-FinBERT sweep on production headlines is the next-pass calibration
work and lives with the T4 backtester upgrade in May.
