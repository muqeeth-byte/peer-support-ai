"""
Main Pipeline — AI-Based Peer Support Recommendation System
"""
import os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.generate_dataset import generate_dataset
from utils.nlp_features import extract_features_batch
from models.cognitive_scorer import score_dataframe, compute_cognitive_scores, generate_need_explanation
from models.peer_matcher import recommend_peers, format_recommendations
from evaluation.evaluator import run_evaluation, print_evaluation_report


def run_pipeline(n_users=500, save_dataset=True, run_eval=True, demo_user_id=None):
    print("\n" + "="*60)
    print("AI-BASED PEER SUPPORT RECOMMENDATION SYSTEM")
    print("="*60)

    dataset_path  = os.path.join(BASE_DIR, "data", "peer_support_dataset.csv")
    enriched_path = os.path.join(BASE_DIR, "data", "peer_support_enriched.csv")

    # Step 1-3: Build or load enriched dataset
    if os.path.exists(enriched_path):
        print(f"\n[1-3] Loading enriched dataset...")
        df = pd.read_csv(enriched_path)
        print(f"      Done. {len(df)} users.")
    else:
        print(f"\n[1] Generating dataset ({n_users} users)...")
        df = generate_dataset(n=n_users, save_path=dataset_path if save_dataset else None)
        print(f"    Done. {len(df)} users.")

        print(f"\n[2] Extracting NLP features...")
        df = extract_features_batch(df)
        print(f"    Done.")

        print(f"\n[3] Running Cognitive Need Scoring...")
        df = score_dataframe(df)
        df.to_csv(enriched_path, index=False)
        print(f"    Done. Saved to {enriched_path}")

    dist = df["PredictedSupport"].value_counts().to_dict()
    print("\nSupport Type Distribution:")
    for k, v in dist.items():
        print(f"  {k:15s}: {v} users ({v/len(df)*100:.1f}%)")

    # Step 4: Demo recommendation
    print("\n[4] Generating sample recommendation...")
    if demo_user_id and demo_user_id in df["UserID"].values:
        user_row = df[df["UserID"] == demo_user_id].iloc[0]
    else:
        candidates = df[df["PredictedSupport"] != "None"]
        user_row = candidates.iloc[0] if len(candidates) else df.iloc[0]

    COL_MAP = {
        "Mood": "mood", "Engagement": "engagement", "EngagementDelta": "engagement_delta",
        "sentiment_compound": "sentiment_compound", "negative_word_freq": "negative_word_freq",
        "academic_signal": "academic_signal", "withdrawal_signal": "withdrawal_signal",
        "motivation_signal": "motivation_signal",
    }
    cog_kwargs = {param: float(user_row[col]) for col, param in COL_MAP.items() if col in user_row.index}
    result = compute_cognitive_scores(**cog_kwargs)
    need_reasons = generate_need_explanation(result)
    recs = recommend_peers(user_row, df, result.support_type, top_k=3)
    report = format_recommendations(user_row["UserID"], result.support_type, need_reasons, recs)
    print(report)

    # Step 5: Evaluation
    if run_eval:
        print("[5] Running Evaluation...")
        results = run_evaluation(df, sample_n=100, top_k=3)
        print_evaluation_report(results)

    print(f"\nPipeline complete.")
    return df


if __name__ == "__main__":
    run_pipeline(n_users=500, save_dataset=True, run_eval=True)
