"""
Peer Matching Engine
MatchScore = w1*D + w2*E + w3*C + w4*A
  D = Domain similarity
  E = Experience match
  C = Communication / skill compatibility (cosine similarity)
  A = Availability overlap (Jaccard)
Returns Top-K ranked peers with explainable factor breakdown.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.nlp_features import (
    encode_skills_vector, encode_availability_vector,
    EXPERIENCE_MAP
)

DEFAULT_WEIGHTS = {"domain": 0.30, "experience": 0.25, "communication": 0.25, "availability": 0.20}

RELATED_GROUPS = [
    {"Computer Science","Data Science","Mathematics","Engineering","Physics"},
    {"Psychology","Education","Literature"},
    {"Biology","Physics"},
    {"Business","Education"},
]


# ── Factor calculators ─────────────────────────────────────────────────────────
def _domain_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    for g in RELATED_GROUPS:
        if a in g and b in g:
            return 0.5
    return 0.0

def _exp_match(ua: str, ub: str) -> float:
    ea, eb = EXPERIENCE_MAP.get(ua, 0), EXPERIENCE_MAP.get(ub, 0)
    diff = eb - ea
    if diff >= 0:
        return [1.0, 0.75, 0.50, 0.30][min(diff, 3)]
    return max(0.0, 0.4 + diff * 0.15)

def _comm_compat(sa: str, sb: str) -> float:
    va = encode_skills_vector(sa).reshape(1, -1)
    vb = encode_skills_vector(sb).reshape(1, -1)
    if va.sum() == 0 or vb.sum() == 0:
        return 0.0
    return round(float(cosine_similarity(va, vb)[0][0]), 4)

def _avail_overlap(aa: str, ab: str) -> float:
    sa = {s.strip() for s in aa.split(",")} if isinstance(aa, str) else set()
    sb = {s.strip() for s in ab.split(",")} if isinstance(ab, str) else set()
    if "Flexible" in sa or "Flexible" in sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return round(inter / union, 4) if union else 0.0


# ── Core matching ──────────────────────────────────────────────────────────────
def compute_match_score(user: pd.Series, peer: pd.Series,
                        weights: Dict = None) -> Tuple[float, Dict]:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    D = _domain_sim(str(user["Domain"]), str(peer["Domain"]))
    E = _exp_match(str(user["Experience"]), str(peer["Experience"]))
    C = _comm_compat(str(user["Skills"]), str(peer["Skills"]))
    A = _avail_overlap(str(user["Availability"]), str(peer["Availability"]))

    total = (weights["domain"]*D + weights["experience"]*E +
             weights["communication"]*C + weights["availability"]*A)
    return round(total, 4), {
        "domain_score": round(D, 4), "experience_score": round(E, 4),
        "communication_score": round(C, 4), "availability_score": round(A, 4),
        "total_score": round(total, 4),
    }


def _explain(peer: pd.Series, f: Dict, support_type: str) -> List[str]:
    reasons = []
    if f["domain_score"] == 1.0:
        reasons.append(f"Same academic domain ({peer['Domain']})")
    elif f["domain_score"] == 0.5:
        reasons.append(f"Related academic domain ({peer['Domain']})")
    if f["experience_score"] >= 0.75:
        reasons.append(f"High experience level ({peer['Experience']}) — ideal for guidance")
    elif f["experience_score"] >= 0.4:
        reasons.append(f"Compatible experience level ({peer['Experience']})")
    if f["communication_score"] >= 0.6:
        reasons.append("Strong skill overlap — good communication compatibility")
    elif f["communication_score"] >= 0.3:
        reasons.append("Moderate skill compatibility")
    if f["availability_score"] >= 0.8:
        reasons.append("Excellent availability overlap")
    elif f["availability_score"] >= 0.4:
        reasons.append(f"Partial availability overlap ({peer['Availability']})")
    tag = {"Emotional": "emotional support conversations",
           "Academic":  "academic tutoring and guidance",
           "Motivational": "motivational coaching"}.get(support_type)
    if tag:
        reasons.append(f"Peer profile well-suited for {tag}")
    return reasons


def recommend_peers(user: pd.Series, peer_pool: pd.DataFrame,
                    support_type: str, top_k: int = 3,
                    weights: Dict = None) -> List[Dict]:
    uid   = user.get("UserID", None)
    pool  = peer_pool[peer_pool["UserID"] != uid] if uid else peer_pool

    scored = []
    for _, peer in pool.iterrows():
        score, factors = compute_match_score(user, peer, weights)
        scored.append({
            "peer_id":           peer["UserID"],
            "peer_domain":       peer["Domain"],
            "peer_experience":   peer["Experience"],
            "peer_skills":       peer["Skills"],
            "peer_availability": peer["Availability"],
            "match_score":       score,
            "factors":           factors,
            "explanation":       _explain(peer, factors, support_type),
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_k]


def format_recommendations(user_id: str, support_type: str,
                            need_reasons: List[str], recommendations: List[Dict]) -> str:
    lines = [
        "="*60,
        "PEER SUPPORT RECOMMENDATION REPORT",
        "="*60,
        f"User         : {user_id}",
        f"Support Need : {support_type}",
        "",
        "Why Support Is Needed:",
    ]
    for r in (need_reasons or ["No specific reasons identified."]):
        lines.append(f"  • {r}")
    lines.append(f"\nTop {len(recommendations)} Recommended Peers:")
    for i, rec in enumerate(recommendations, 1):
        lines += [
            f"\n  Rank #{i}: {rec['peer_id']}  (Match Score: {rec['match_score']:.4f})",
            f"    Domain Similarity    : {rec['factors']['domain_score']:.2f}",
            f"    Experience Match     : {rec['factors']['experience_score']:.2f}",
            f"    Communication Compat : {rec['factors']['communication_score']:.2f}",
            f"    Availability Overlap : {rec['factors']['availability_score']:.2f}",
            f"  Why this peer:",
        ]
        for r in rec["explanation"]:
            lines.append(f"    → {r}")
    lines.append("\n" + "="*60)
    return "\n".join(lines)


if __name__ == "__main__":
    user = pd.Series({"UserID":"U001","Domain":"Computer Science",
                      "Skills":"Python, Machine Learning","Experience":"Beginner",
                      "Availability":"Evening, Weekends"})
    pool = pd.DataFrame([
        {"UserID":"U002","Domain":"Computer Science","Skills":"Python, Machine Learning, Data Analysis",
         "Experience":"Advanced","Availability":"Evening"},
        {"UserID":"U003","Domain":"Data Science","Skills":"Statistics, Machine Learning",
         "Experience":"Intermediate","Availability":"Weekends"},
        {"UserID":"U004","Domain":"Biology","Skills":"Writing, Research",
         "Experience":"Expert","Availability":"Morning"},
    ])
    recs = recommend_peers(user, pool, "Academic", top_k=3)
    print(format_recommendations("U001","Academic",["Low engagement detected"],recs))
