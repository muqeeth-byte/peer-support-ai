"""
feature_engineering.py
-----------------------
NLP + feature transformation pipeline.

Converts raw user inputs into structured, numeric feature vectors:
  - Sentiment polarity from text reflection (TextBlob)
  - Negative word frequency count
  - Engagement delta (current vs previous session)
  - Domain encoding (label encoding)
  - Skill vector (multi-hot)
  - Availability encoding (multi-hot)
  - Experience level encoding (ordinal)
"""

import re
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# ── Constants ──────────────────────────────────────────────────────────────────
NEGATIVE_WORDS = {
    "overwhelmed", "struggling", "lost", "anxious", "exhausted",
    "demotivated", "isolated", "frustrated", "hopeless", "absent",
    "withdrawn", "unable", "cannot", "stressed", "confused",
    "behind", "failing", "difficult", "hard", "stuck",
    "worried", "nervous", "sad", "unhappy", "depressed"
}

ALL_SKILLS = sorted([
    "Python", "Statistics", "Writing", "Leadership", "Research",
    "Communication", "Critical Thinking", "Problem Solving",
    "Data Analysis", "Teaching", "Mentoring", "Machine Learning",
    "Mathematics", "Project Management", "Public Speaking"
])

ALL_SLOTS = sorted([
    "Morning", "Afternoon", "Evening", "Night",
    "Weekends", "Weekdays", "Flexible"
])

EXPERIENCE_MAP = {"Beginner": 1, "Intermediate": 2, "Advanced": 3, "Expert": 4}

ALL_DOMAINS = sorted([
    "Computer Science", "Psychology", "Engineering", "Biology",
    "Mathematics", "Business", "Education", "Physics",
    "Data Science", "Literature"
])


# ── Text utilities ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase and strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def get_sentiment_polarity(text: str) -> float:
    """Return TextBlob polarity in [-1.0, +1.0]."""
    return TextBlob(text).sentiment.polarity


def get_negative_word_freq(text: str) -> int:
    """Count how many negative keywords appear in the text."""
    tokens = set(clean_text(text).split())
    return len(tokens & NEGATIVE_WORDS)


# ── Feature builders ───────────────────────────────────────────────────────────
def skill_vector(skills_str: str) -> np.ndarray:
    """Multi-hot vector over ALL_SKILLS."""
    skills = {s.strip() for s in skills_str.split("|")}
    return np.array([1 if s in skills else 0 for s in ALL_SKILLS], dtype=float)


def availability_vector(avail_str: str) -> np.ndarray:
    """Multi-hot vector over ALL_SLOTS."""
    slots = {s.strip() for s in avail_str.split("|")}
    return np.array([1 if s in slots else 0 for s in ALL_SLOTS], dtype=float)


def domain_index(domain: str) -> int:
    """0-based index into ALL_DOMAINS list."""
    return ALL_DOMAINS.index(domain) if domain in ALL_DOMAINS else -1


def experience_score(level: str) -> int:
    return EXPERIENCE_MAP.get(level, 1)


def engagement_delta(current: float, previous: float) -> float:
    return round(current - previous, 4)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw user DataFrame and returns an enriched DataFrame
    with all engineered feature columns appended.
    """
    df = df.copy()

    # Text features
    df["SentimentPolarity"] = df["Reflection"].apply(get_sentiment_polarity)
    df["NegWordFreq"]       = df["Reflection"].apply(get_negative_word_freq)

    # Engagement delta
    df["EngagementDelta"] = df.apply(
        lambda r: engagement_delta(r["Engagement"], r["PrevEngagement"]), axis=1
    )

    # Ordinal encodings
    df["ExperienceScore"] = df["ExperienceLevel"].apply(experience_score)
    df["DomainIndex"]     = df["Domain"].apply(domain_index)

    # Skill and availability vectors stored as list columns
    df["SkillVector"]        = df["Skills"].apply(skill_vector)
    df["AvailabilityVector"] = df["Availability"].apply(availability_vector)

    return df


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generate_dataset import generate_dataset

    df_raw = generate_dataset(10)
    df_feat = engineer_features(df_raw)

    print("Engineered feature columns added:")
    new_cols = ["SentimentPolarity","NegWordFreq","EngagementDelta",
                "ExperienceScore","DomainIndex"]
    print(df_feat[new_cols].to_string())
    print("\nSkill vector sample:", df_feat["SkillVector"].iloc[0])
