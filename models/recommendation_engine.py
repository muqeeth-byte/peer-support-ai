"""
recommendation_engine.py
-------------------------
Top-level facade that wires together the full pipeline:

  Input Data  →  Feature Engineering  →  Cognitive Scoring
              →  Peer Matching         →  Explainable Output
"""

import pandas as pd
from typing import Optional, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.feature_engineering import (
    engineer_features, get_sentiment_polarity, get_negative_word_freq,
    engagement_delta, skill_vector, availability_vector,
    domain_index, experience_score, ALL_SKILLS, ALL_SLOTS
)
from models.cognitive_scorer import compute_cognitive_scores, generate_need_explanation
from models.peer_matcher import recommend_peers, format_recommendations


def build_user_row(
    user_id: str,
    mood: int,
    engagement: float,
    prev_engagement: float,
    reflection: str,
    domain: str,
    skills: str,
    availability: str,
    experience: str,
) -> pd.Series:
    """
    Build a feature-rich pd.Series for a single user from raw inputs.
    """
    sentiment  = get_sentiment_polarity(reflection)
    neg_freq   = get_negative_word_freq(reflection)
    eng_delta  = engagement_delta(engagement, prev_engagement)
    exp_score  = experience_score(experience)
    dom_index  = domain_index(domain)
    skill_vec  = skill_vector(skills)
    avail_vec  = availability_vector(availability)

    return pd.Series({
        "UserID":            user_id,
        "Mood":              mood,
        "Engagement":        engagement,
        "PrevEngagement":    prev_engagement,
        "Reflection":        reflection,
        "Domain":            domain,
        "Skills":            skills,
        "Availability":      availability,
        "ExperienceLevel":   experience,
        "Profile":           "unknown",
        # Engineered
        "SentimentPolarity": sentiment,
        "NegWordFreq":       neg_freq,
        "EngagementDelta":   eng_delta,
        "ExperienceScore":   exp_score,
        "DomainIndex":       dom_index,
        "SkillVector":       skill_vec,
        "AvailabilityVector":avail_vec,
    })


def run_pipeline(
    user_row: pd.Series,
    peer_pool: pd.DataFrame,
    top_k: int = 3,
    weights: Optional[Dict] = None,
    return_text: bool = True,
):
    """
    Execute the full recommendation pipeline for a single user.

    Returns
    -------
    (cognitive_result, recommendations, explanation_text)
    """
    # Step 1: Cognitive Need Scoring
    cog = compute_cognitive_scores(
        mood               = user_row["Mood"],
        engagement         = user_row["Engagement"],
        engagement_delta   = user_row["EngagementDelta"],
        sentiment_polarity = user_row["SentimentPolarity"],
        neg_word_freq      = user_row["NegWordFreq"],
    )

    # Step 2: Peer Matching
    recs = recommend_peers(user_row, peer_pool, top_k=top_k, weights=weights)

    # Step 3: Explainable Output
    text = ""
    if return_text:
        need_explanation = "\n".join(generate_need_explanation(cog))
        rec_explanation  = format_recommendations(
            recs,
            support_type  = cog.support_type,
            query_reasons = cog.reasons.get(cog.support_type, []),
        )
        text = need_explanation + "\n\n" + rec_explanation

    return cog, recs, text


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generate_dataset import generate_dataset

    df_full = engineer_features(generate_dataset(100))

    query_user = build_user_row(
        user_id       = "TestUser",
        mood          = 3,
        engagement    = 0.25,
        prev_engagement = 0.45,
        reflection    = "I feel overwhelmed and cannot keep up with my studies.",
        domain        = "Computer Science",
        skills        = "Python|Statistics",
        availability  = "Evening|Weekends",
        experience    = "Intermediate",
    )

    cog, recs, text = run_pipeline(query_user, df_full, top_k=3)
    print(text)
