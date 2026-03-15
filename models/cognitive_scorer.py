"""
Cognitive Need Scoring Engine
Rule-based explainable model detecting support needs: Emotional / Academic / Motivational.
All scoring is transparent — no black-box models.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

# ── Thresholds ─────────────────────────────────────────────────────────────────
LOW_MOOD_THRESHOLD       = 4.0
MID_MOOD_THRESHOLD       = 7.0
LOW_ENGAGEMENT_THRESHOLD = 0.40
MID_ENGAGEMENT_THRESHOLD = 0.60
NEGATIVE_DELTA_THRESHOLD = -0.10
SENTIMENT_NEG_THRESHOLD  = -0.10
SENTIMENT_NEU_THRESHOLD  =  0.10


@dataclass
class CognitiveNeedResult:
    emotional_score:    float = 0.0
    academic_score:     float = 0.0
    motivational_score: float = 0.0
    support_type:       str   = "None"
    confidence:         float = 0.0
    reasons:            Dict[str, List[str]] = field(default_factory=dict)


def compute_cognitive_scores(
    mood: float,
    engagement: float,
    engagement_delta: float,
    sentiment_compound: float = 0.0,
    negative_word_freq: float = 0.0,
    academic_signal: float = 0.0,
    withdrawal_signal: float = 0.0,
    motivation_signal: float = 0.0,
    **kwargs
) -> CognitiveNeedResult:
    """
    Core scoring function. All parameters are floats from the feature layer.
    **kwargs absorbs any extra DataFrame columns safely.
    """
    result = CognitiveNeedResult()
    reasons: Dict[str, List[str]] = {"Emotional": [], "Academic": [], "Motivational": []}

    # ── EMOTIONAL ──────────────────────────────────────────────────────────────
    if mood <= LOW_MOOD_THRESHOLD:
        result.emotional_score += 3.0
        reasons["Emotional"].append(f"Low mood score ({mood}/10)")

    if sentiment_compound < SENTIMENT_NEG_THRESHOLD:
        result.emotional_score += 2.0
        reasons["Emotional"].append(f"Negative text sentiment (compound: {sentiment_compound:.2f})")

    neg_count = negative_word_freq * 20  # reverse normalise for rule thresholding
    if neg_count >= 3:
        result.emotional_score += 2.0
        reasons["Emotional"].append("Multiple distress keywords detected in reflection")
    elif neg_count >= 1:
        result.emotional_score += 1.0
        reasons["Emotional"].append("Distress keywords present in reflection")

    if withdrawal_signal > 0.05:
        result.emotional_score += 1.5
        reasons["Emotional"].append("Withdrawal language detected in reflection")

    if engagement_delta < NEGATIVE_DELTA_THRESHOLD and engagement < 0.35:
        result.emotional_score += 1.5
        reasons["Emotional"].append("Sudden withdrawal from engagement")

    # ── ACADEMIC ───────────────────────────────────────────────────────────────
    if engagement <= LOW_ENGAGEMENT_THRESHOLD:
        result.academic_score += 3.0
        reasons["Academic"].append(f"Low engagement score ({engagement:.0%})")

    if engagement_delta < NEGATIVE_DELTA_THRESHOLD:
        result.academic_score += 2.0
        reasons["Academic"].append(f"Engagement declined by {abs(engagement_delta):.0%}")

    if academic_signal > 0.05:
        result.academic_score += 2.0
        reasons["Academic"].append("Academic difficulty keywords detected in reflection")

    if mood >= 5 and engagement <= LOW_ENGAGEMENT_THRESHOLD:
        result.academic_score += 1.0
        reasons["Academic"].append("Mood adequate but engagement low — possible academic barrier")

    # ── MOTIVATIONAL ───────────────────────────────────────────────────────────
    if LOW_MOOD_THRESHOLD < mood <= MID_MOOD_THRESHOLD:
        result.motivational_score += 2.0
        reasons["Motivational"].append(f"Neutral mood zone ({mood}/10)")

    if LOW_ENGAGEMENT_THRESHOLD < engagement <= MID_ENGAGEMENT_THRESHOLD:
        result.motivational_score += 2.0
        reasons["Motivational"].append(f"Moderate engagement plateau ({engagement:.0%})")

    if SENTIMENT_NEG_THRESHOLD <= sentiment_compound <= SENTIMENT_NEU_THRESHOLD:
        result.motivational_score += 1.0
        reasons["Motivational"].append("Neutral/flat reflective tone")

    if motivation_signal > 0.05:
        result.motivational_score += 1.5
        reasons["Motivational"].append("Motivational keywords detected in reflection")

    if NEGATIVE_DELTA_THRESHOLD <= engagement_delta < 0:
        result.motivational_score += 1.0
        reasons["Motivational"].append(f"Slight engagement decline ({engagement_delta:+.0%})")

    # ── Classify ────────────────────────────────────────────────────────────────
    scores = {
        "Emotional":    result.emotional_score,
        "Academic":     result.academic_score,
        "Motivational": result.motivational_score,
    }
    best = max(scores, key=scores.get)
    max_score = scores[best]

    # Threshold: if top score < 2.0, user is doing fine
    if max_score < 2.0:
        result.support_type = "None"
        result.confidence = 0.0
    else:
        result.support_type = best
        # Normalize confidence: max possible ≈ 10
        result.confidence = round(min(max_score / 10.0, 1.0), 4)

    result.reasons = reasons
    return result


def generate_need_explanation(result: CognitiveNeedResult) -> List[str]:
    """Return list of human-readable reasons for the detected support need."""
    if result.support_type == "None":
        return ["User appears to be performing well — no urgent support needed."]
    return result.reasons.get(result.support_type, [])


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cognitive scoring to entire DataFrame. Returns enriched DataFrame."""
    COL_MAP = {
        "Mood": "mood", "Engagement": "engagement",
        "EngagementDelta": "engagement_delta",
        "sentiment_compound": "sentiment_compound",
        "negative_word_freq": "negative_word_freq",
        "academic_signal": "academic_signal",
        "withdrawal_signal": "withdrawal_signal",
        "motivation_signal": "motivation_signal",
    }
    rows = []
    for _, row in df.iterrows():
        kwargs = {param: float(row[col]) for col, param in COL_MAP.items() if col in row.index}
        result = compute_cognitive_scores(**kwargs)
        rows.append({
            "emotional_score":    round(result.emotional_score, 4),
            "academic_score":     round(result.academic_score, 4),
            "motivational_score": round(result.motivational_score, 4),
            "PredictedSupport":   result.support_type,
            "SupportConfidence":  result.confidence,
        })
    score_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), score_df], axis=1)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        dict(mood=2.0, engagement=0.20, engagement_delta=-0.18,
             sentiment_compound=-0.65, negative_word_freq=0.25,
             academic_signal=0.0, withdrawal_signal=0.15, motivation_signal=0.0),
        dict(mood=5.5, engagement=0.30, engagement_delta=-0.20,
             sentiment_compound=-0.10, negative_word_freq=0.10,
             academic_signal=0.20, withdrawal_signal=0.0, motivation_signal=0.0),
        dict(mood=6.0, engagement=0.50, engagement_delta=-0.05,
             sentiment_compound=0.0, negative_word_freq=0.05,
             academic_signal=0.05, withdrawal_signal=0.0, motivation_signal=0.15),
        dict(mood=8.5, engagement=0.90, engagement_delta=0.10,
             sentiment_compound=0.60, negative_word_freq=0.0,
             academic_signal=0.0, withdrawal_signal=0.0, motivation_signal=0.0),
    ]
    labels = ["Emotional User", "Academic User", "Motivational User", "Balanced User"]
    for label, c in zip(labels, cases):
        res = compute_cognitive_scores(**c)
        print(f"\n[{label}]")
        print(f"  Emotional={res.emotional_score:.1f}  Academic={res.academic_score:.1f}  Motivational={res.motivational_score:.1f}")
        print(f"  → {res.support_type} (confidence: {res.confidence:.2%})")
        for r in generate_need_explanation(res):
            print(f"    • {r}")
