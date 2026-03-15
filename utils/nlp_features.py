"""
NLP Feature Extraction Module
Transforms raw text reflections into quantifiable behavioral indicators.
Falls back gracefully when VADER/NLTK data is unavailable (network-restricted environments).
"""

import re
import numpy as np
import pandas as pd
from typing import Dict

# ── NLTK with graceful fallback ────────────────────────────────────────────────
import nltk
for _r in ["vader_lexicon", "stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
    _VADER_OK = True
except Exception:
    _VADER_OK = False

try:
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words("english"))
except Exception:
    STOP_WORDS = {
        "i","me","my","we","our","you","your","he","him","his","she","her",
        "they","them","their","what","which","who","this","that","these","those",
        "am","is","are","was","were","be","been","have","has","had","do","does",
        "did","a","an","the","and","but","if","or","as","at","by","for","of",
        "on","to","up","in","out","it","its","so","not","no","how","all","both",
        "each","few","more","most","other","some","such","own","same","than",
        "then","there","when","where","why","will","with","about","after",
        "before","into","through","over","any","can","could","would","should",
        "may","might","must","now","only","also","just","very","too",
    }

# ── Keyword lexicons ───────────────────────────────────────────────────────────
NEGATIVE_WORDS = {
    "overwhelmed","exhausted","hopeless","anxious","stressed","lonely",
    "isolated","drained","burnt","struggling","failing","lost","confused",
    "stuck","sad","depressed","frustrated","worried","unable","difficult",
    "hard","impossible","pointless","worthless","withdrawn","disconnected",
}
ACADEMIC_WORDS = {
    "failing","grade","assignment","exam","study","understand","concepts",
    "coursework","deadline","behind","methodology","research","material",
    "subject","learn","confused","formula","approach","knowledge","topic",
}
WITHDRAWAL_WORDS = {
    "alone","withdrawn","disconnected","isolated","avoiding","hiding",
    "antisocial","escape","retreat","lonely","distance",
}
MOTIVATION_WORDS = {
    "procrastinating","unmotivated","lazy","stuck","delay","accountable",
    "consistent","drive","energy","inspire","execute","goals","push",
}

EXPERIENCE_MAP = {"Beginner": 0, "Intermediate": 1, "Advanced": 2, "Expert": 3}
DOMAIN_INDEX   = {
    "Computer Science": 0, "Mathematics": 1, "Psychology": 2, "Engineering": 3,
    "Biology": 4, "Business": 5, "Literature": 6, "Physics": 7,
    "Data Science": 8, "Education": 9,
}
ALL_SKILLS = [
    "Python","Statistics","Writing","Research","Communication","Problem Solving",
    "Data Analysis","Leadership","Critical Thinking","Machine Learning",
    "Project Management","Presentation","Programming","Teamwork","Time Management",
]
AVAILABILITY_OPTIONS = ["Morning","Afternoon","Evening","Night","Weekends","Flexible"]


# ── Text helpers ───────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _tokens(text: str):
    return [w for w in _clean(text).split() if w not in STOP_WORDS and len(w) > 1]

def _kwfrac(tokens, lexicon) -> float:
    n = max(len(tokens), 1)
    return round(sum(1 for t in tokens if t in lexicon) / n, 4)

def _sentiment(text: str) -> Dict[str, float]:
    if not text:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    if _VADER_OK:
        return _SIA.polarity_scores(text)
    # Heuristic fallback
    toks = _tokens(text)
    nf   = _kwfrac(toks, NEGATIVE_WORDS)
    comp = max(-1.0, min(1.0, round(-nf * 4.0, 4)))
    return {"neg": min(nf, 1.0), "neu": max(0.0, 1.0 - nf * 2), "pos": 0.0, "compound": comp}


# ── Main extraction ────────────────────────────────────────────────────────────
def extract_features(text: str) -> Dict[str, float]:
    """Extract NLP features from a single reflection string."""
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in [
            "sentiment_compound","sentiment_neg","sentiment_pos",
            "negative_word_freq","academic_signal","withdrawal_signal",
            "motivation_signal","text_length_norm",
        ]}
    sent  = _sentiment(text)
    toks  = _tokens(text)
    return {
        "sentiment_compound":  sent["compound"],
        "sentiment_neg":       sent["neg"],
        "sentiment_pos":       sent["pos"],
        "negative_word_freq":  _kwfrac(toks, NEGATIVE_WORDS),
        "academic_signal":     _kwfrac(toks, ACADEMIC_WORDS),
        "withdrawal_signal":   _kwfrac(toks, WITHDRAWAL_WORDS),
        "motivation_signal":   _kwfrac(toks, MOTIVATION_WORDS),
        "text_length_norm":    round(min(len(toks) / 30.0, 1.0), 4),
    }


def extract_features_batch(df: pd.DataFrame, text_col: str = "Reflection") -> pd.DataFrame:
    """Append NLP feature columns to a DataFrame."""
    feats = df[text_col].apply(extract_features).tolist()
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


# ── Encoding helpers (used by peer_matcher) ────────────────────────────────────
def encode_skills_vector(skills_str: str) -> np.ndarray:
    vec = np.zeros(len(ALL_SKILLS))
    if isinstance(skills_str, str):
        user_skills = {s.strip() for s in skills_str.split(",")}
        for i, s in enumerate(ALL_SKILLS):
            if s in user_skills:
                vec[i] = 1.0
    return vec

def encode_availability_vector(avail_str: str) -> np.ndarray:
    vec = np.zeros(len(AVAILABILITY_OPTIONS))
    if isinstance(avail_str, str):
        slots = {s.strip() for s in avail_str.split(",")}
        for i, opt in enumerate(AVAILABILITY_OPTIONS):
            if opt in slots:
                vec[i] = 1.0
    return vec

def encode_experience(exp: str) -> int:
    return EXPERIENCE_MAP.get(exp, 0)


if __name__ == "__main__":
    tests = [
        "I feel overwhelmed and exhausted. I can't cope anymore.",
        "Struggling to understand the mathematical concepts in my exam.",
        "I know what to do but can't start. I keep procrastinating.",
        "Things are going well today. Feeling productive.",
        "",
    ]
    for t in tests:
        f = extract_features(t)
        print(f"\n'{t[:55]}...'")
        for k, v in f.items():
            print(f"  {k:<25}: {v}")
