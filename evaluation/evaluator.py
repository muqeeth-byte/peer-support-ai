"""
Evaluation Module — Precision@K, Matching Accuracy, FPR, User Satisfaction
Compares AI system vs. random baseline.
"""

import os, sys, random
import numpy as np
import pandas as pd
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.nlp_features import EXPERIENCE_MAP
from models.peer_matcher import recommend_peers

RELATED_GROUPS = [
    {"Computer Science", "Data Science", "Mathematics", "Engineering", "Physics"},
    {"Psychology", "Education", "Literature"},
    {"Biology", "Physics"},
    {"Business", "Education"},
]

def is_relevant_peer(user: pd.Series, peer: pd.Series) -> bool:
    """A peer is 'relevant' if same/related domain AND experience >= user."""
    if peer["UserID"] == user["UserID"]:
        return False
    user_exp  = EXPERIENCE_MAP.get(user["Experience"], 0)
    peer_exp  = EXPERIENCE_MAP.get(peer["Experience"], 0)
    same_dom  = peer["Domain"] == user["Domain"]
    rel_dom   = any(peer["Domain"] in g and user["Domain"] in g for g in RELATED_GROUPS)
    return (same_dom or rel_dom) and peer_exp >= user_exp


def precision_at_k(recommended_ids: List[str], relevant_ids: set, k: int = 3) -> float:
    hits = sum(1 for pid in recommended_ids[:k] if pid in relevant_ids)
    return round(hits / k, 4)


def matching_accuracy(df: pd.DataFrame) -> float:
    if "TrueSupport" not in df.columns or "PredictedSupport" not in df.columns:
        return 0.0
    return round((df["TrueSupport"] == df["PredictedSupport"]).mean(), 4)


def false_positive_rate(df: pd.DataFrame) -> float:
    no_support = df[df["TrueSupport"] == "None"]
    if len(no_support) == 0:
        return 0.0
    fp = (no_support["PredictedSupport"] != "None").sum()
    tn = (no_support["PredictedSupport"] == "None").sum()
    denom = fp + tn
    return round(fp / denom, 4) if denom > 0 else 0.0


def random_precision_at_k(user: pd.Series, peer_pool: pd.DataFrame,
                           relevant_ids: set, k: int = 3, trials: int = 100) -> float:
    candidates = peer_pool[peer_pool["UserID"] != user["UserID"]]["UserID"].tolist()
    if len(candidates) < k:
        return 0.0
    scores = []
    for _ in range(trials):
        picks = random.sample(candidates, k)
        scores.append(sum(1 for p in picks if p in relevant_ids) / k)
    return round(np.mean(scores), 4)


def run_evaluation(df: pd.DataFrame, sample_n: int = 100, top_k: int = 3, seed: int = 42) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    support_users = df[df["PredictedSupport"] != "None"].copy()
    if len(support_users) > sample_n:
        support_users = support_users.sample(n=sample_n, random_state=seed)

    sys_precisions, rand_precisions, all_scores = [], [], []

    for _, user in support_users.iterrows():
        relevant = {peer["UserID"] for _, peer in df.iterrows() if is_relevant_peer(user, peer)}
        if len(relevant) < top_k:
            continue

        recs = recommend_peers(user, df, user["PredictedSupport"], top_k=top_k)
        rec_ids = [r["peer_id"] for r in recs]
        rec_scores = [r["match_score"] for r in recs]

        sys_precisions.append(precision_at_k(rec_ids, relevant, k=top_k))
        rand_precisions.append(random_precision_at_k(user, df, relevant, k=top_k))
        all_scores.extend(rec_scores)

    satisfaction = round(sum(1 for s in all_scores if s >= 0.5) / max(len(all_scores), 1), 4)

    return {
        "matching_accuracy":       matching_accuracy(df),
        "false_positive_rate":     false_positive_rate(df),
        "precision_at_k_system":   round(np.mean(sys_precisions), 4)  if sys_precisions  else 0.0,
        "precision_at_k_random":   round(np.mean(rand_precisions), 4) if rand_precisions else 0.0,
        "user_satisfaction_score": satisfaction,
        "avg_match_score":         round(np.mean(all_scores), 4) if all_scores else 0.0,
        "users_evaluated":         len(sys_precisions),
    }


def print_evaluation_report(results: Dict):
    print("\n" + "="*60)
    print("EVALUATION REPORT — AI PEER SUPPORT SYSTEM")
    print("="*60)
    print(f"  Users Evaluated         : {results['users_evaluated']}")
    print(f"\n  Classification Metrics:")
    print(f"    Matching Accuracy     : {results['matching_accuracy']:.2%}")
    print(f"    False Positive Rate   : {results['false_positive_rate']:.2%}")
    print(f"\n  Recommendation Metrics:")
    print(f"    Precision@K (System)  : {results['precision_at_k_system']:.4f}")
    print(f"    Precision@K (Random)  : {results['precision_at_k_random']:.4f}")
    imp = results['precision_at_k_system'] - results['precision_at_k_random']
    print(f"    Improvement           : +{imp:.4f}")
    print(f"\n  Quality Metrics:")
    print(f"    Avg Match Score       : {results['avg_match_score']:.4f}")
    print(f"    User Satisfaction     : {results['user_satisfaction_score']:.2%}")
    print("="*60)


if __name__ == "__main__":
    from data.generate_dataset import generate_dataset
    from utils.nlp_features import extract_features_batch
    from models.cognitive_scorer import score_dataframe
    df = generate_dataset(n=500)
    df = extract_features_batch(df)
    df = score_dataframe(df)
    results = run_evaluation(df, sample_n=100)
    print_evaluation_report(results)
