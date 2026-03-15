"""
app.py  —  Streamlit UI for the AI-Based Peer Support Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_dataset import generate_dataset
from utils.feature_engineering import engineer_features
from models.recommendation_engine import build_user_row, run_pipeline
from models.cognitive_scorer import CognitiveNeedResult
from models.peer_matcher import PeerMatch
from evaluation.evaluator import evaluate

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Peer Support",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 18px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}
.peer-card {
    background: #f8f9fa;
    border-left: 5px solid #667eea;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
}
.support-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.9em;
}
.emotional   { background: #ff6b6b; color: white; }
.academic    { background: #4ecdc4; color: white; }
.motivational{ background: #f9ca24; color: #2d3436; }
</style>
""", unsafe_allow_html=True)

# ── Caching ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_peer_pool():
    df = generate_dataset(520)
    return engineer_features(df)

@st.cache_data
def run_eval():
    return evaluate(n_users=100, top_k=3)

PEER_POOL = load_peer_pool()

DOMAINS = sorted([
    "Computer Science", "Psychology", "Engineering", "Biology",
    "Mathematics", "Business", "Education", "Physics",
    "Data Science", "Literature"
])
SKILLS = sorted([
    "Python", "Statistics", "Writing", "Leadership", "Research",
    "Communication", "Critical Thinking", "Problem Solving",
    "Data Analysis", "Teaching", "Mentoring", "Machine Learning",
    "Mathematics", "Project Management", "Public Speaking"
])
SLOTS = ["Morning", "Afternoon", "Evening", "Night", "Weekends", "Weekdays", "Flexible"]
EXP   = ["Beginner", "Intermediate", "Advanced", "Expert"]
SUPPORT_COLORS = {"Emotional": "emotional", "Academic": "academic", "Motivational": "motivational"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/handshake.png", width=80)
    st.title("🤝 Peer Support AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Get Recommendations", "📊 Evaluation Dashboard", "ℹ️ About"]
    )
    st.markdown("---")
    st.subheader("⚙️ Matching Weights")
    w_domain = st.slider("Domain Weight",       0.0, 1.0, 0.30, 0.05)
    w_exp    = st.slider("Experience Weight",   0.0, 1.0, 0.25, 0.05)
    w_comm   = st.slider("Skill/Comm Weight",   0.0, 1.0, 0.25, 0.05)
    w_avail  = st.slider("Availability Weight", 0.0, 1.0, 0.20, 0.05)
    total_w  = w_domain + w_exp + w_comm + w_avail
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f}. Best if they sum to 1.0")

# ── Page: Get Recommendations ──────────────────────────────────────────────────
if page == "🔍 Get Recommendations":
    st.title("🔍 AI-Based Peer Support Recommendation")
    st.markdown(
        "Complete the check-in form below. The system will detect your support needs "
        "and recommend the top 3 peers best suited to help you."
    )

    with st.form("checkin_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_id    = st.text_input("Your User ID (optional)", value="Guest_001")
            mood       = st.slider("Mood Score (1 = very low, 10 = excellent)", 1, 10, 5)
            engagement = st.slider("Engagement Level (0 = disengaged, 1 = fully engaged)",
                                   0.0, 1.0, 0.5, 0.05)
            prev_eng   = st.slider("Previous Session Engagement", 0.0, 1.0, 0.5, 0.05)
            domain     = st.selectbox("Academic Domain", DOMAINS)

        with col2:
            experience   = st.selectbox("Experience Level", EXP)
            skills_sel   = st.multiselect("Skill Strengths", SKILLS,
                                          default=["Communication", "Problem Solving"])
            avail_sel    = st.multiselect("Availability", SLOTS,
                                          default=["Evening", "Weekends"])
            reflection   = st.text_area(
                "Optional: How are you feeling today? (text reflection)",
                placeholder="Describe your current state, challenges, or goals...",
                height=120,
            )
            top_k = st.selectbox("Number of Peers to Recommend", [1, 2, 3, 5], index=2)

        submitted = st.form_submit_button("🤝 Find My Peers", use_container_width=True)

    if submitted:
        if not skills_sel:
            st.error("Please select at least one skill.")
        elif not avail_sel:
            st.error("Please select at least one availability slot.")
        else:
            skills_str = "|".join(skills_sel)
            avail_str  = "|".join(avail_sel)
            if not reflection.strip():
                reflection = "No reflection provided."

            weights = {
                "domain": w_domain, "experience": w_exp,
                "communication": w_comm, "availability": w_avail,
            }

            with st.spinner("Analysing your profile and finding best peers..."):
                user_row = build_user_row(
                    user_id=user_id, mood=mood,
                    engagement=engagement, prev_engagement=prev_eng,
                    reflection=reflection, domain=domain,
                    skills=skills_str, availability=avail_str,
                    experience=experience,
                )
                cog, recs, _ = run_pipeline(
                    user_row, PEER_POOL, top_k=top_k, weights=weights, return_text=False
                )

            # ── Results ───────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Your Support Profile")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mood", f"{mood}/10")
            c2.metric("Engagement", f"{engagement:.0%}")
            c3.metric("Engagement Δ", f"{engagement - prev_eng:+.0%}")
            c4.metric("Sentiment", f"{user_row['SentimentPolarity']:.2f}")

            st.markdown("---")

            # Support need display
            badge_class = SUPPORT_COLORS.get(cog.support_type, "academic")
            st.markdown(
                f"**Detected Support Need:** "
                f"<span class='support-badge {badge_class}'>{cog.support_type}</span>",
                unsafe_allow_html=True
            )

            score_df = pd.DataFrame({
                "Support Type": ["Emotional", "Academic", "Motivational"],
                "Score": [cog.emotional_score, cog.academic_score, cog.motivational_score],
            }).set_index("Support Type")
            st.bar_chart(score_df)

            reasons = cog.reasons.get(cog.support_type, [])
            if reasons:
                st.markdown("**Why this support type was identified:**")
                for r in reasons:
                    st.markdown(f"  • {r}")

            st.markdown("---")
            st.subheader(f"🏆 Top {top_k} Peer Recommendations")

            for rank, peer in enumerate(recs, 1):
                with st.container():
                    st.markdown(
                        f"""<div class='peer-card'>
                        <b>Rank {rank}: {peer.peer_id}</b> &nbsp;
                        <span style='color:#667eea;'>MatchScore: {peer.match_score:.3f}</span>
                        <br><b>Domain:</b> {peer.peer_domain} &nbsp;|&nbsp;
                        <b>Experience:</b> {peer.peer_experience}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Domain (D)", f"{peer.domain_score:.2f}")
                    fc2.metric("Experience (E)", f"{peer.experience_score:.2f}")
                    fc3.metric("Skills (C)", f"{peer.comm_score:.2f}")
                    fc4.metric("Availability (A)", f"{peer.avail_score:.2f}")

                    if peer.reasons:
                        st.markdown("**Match Reasons:**")
                        for r in peer.reasons:
                            st.markdown(f"  ✅ {r}")
                    st.markdown("---")

# ── Page: Evaluation Dashboard ─────────────────────────────────────────────────
elif page == "📊 Evaluation Dashboard":
    st.title("📊 System Evaluation Dashboard")
    st.markdown(
        "Performance metrics evaluated on 100 randomly sampled users "
        "compared against random peer baseline."
    )
    with st.spinner("Running evaluation..."):
        results = run_eval()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@K (System)", f"{results['Precision@K (System)']:.3f}")
    c2.metric("Precision@K (Random)", f"{results['Precision@K (Random)']:.3f}")
    c3.metric("Matching Accuracy",    f"{results['Matching Accuracy']:.3f}")
    c4.metric("User Satisfaction",    f"{results['User Satisfaction']:.3f}")

    st.metric("False Positive Rate", f"{results['False Positive Rate']:.4f}")
    improvement = results["Improvement over Random"]
    st.success(f"✅ System outperforms random baseline by **{improvement:.4f}** in Precision@K")

    st.markdown("---")
    st.subheader("Score Comparison Chart")
    compare_df = pd.DataFrame({
        "Metric": ["Precision@K", "Matching Accuracy"],
        "System":  [results["Precision@K (System)"], results["Matching Accuracy"]],
        "Random":  [results["Precision@K (Random)"], 0.5],
    }).set_index("Metric")
    st.bar_chart(compare_df)

# ── Page: About ────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("""
## AI-Based Peer Support Recommendation System
**A Cognitive Analytics Framework for Intelligent Peer Matching**

### System Architecture (5 Layers)
1. **User Input Layer** — Collects mood, engagement, reflection text, domain, skills, availability, experience
2. **Feature Engineering Layer** — Sentiment analysis (TextBlob), negative word frequency, engagement delta, skill vectors, availability vectors
3. **Cognitive Need Scoring Engine** — Rule-based transparent scoring for Emotional / Academic / Motivational support
4. **Peer Matching Engine** — Weighted similarity scoring: `MatchScore = w₁D + w₂E + w₃C + w₄A`
5. **Recommendation Output Layer** — Top-K ranked peers with explainable reasoning

### Matching Formula
```
MatchScore = w₁ × Domain + w₂ × Experience + w₃ × Skills + w₄ × Availability
```
Weights are fully adjustable in the sidebar.

### Evaluation Metrics
- **Precision@K** — Fraction of top-K recommendations that are truly relevant
- **Matching Accuracy** — Percentage of users with at least one relevant match in top-K
- **User Satisfaction Score** — Proxy score based on average match quality
- **False Positive Rate** — Fraction of irrelevant recommendations

### Tech Stack
| Component | Technology |
|-----------|------------|
| Backend   | Python 3   |
| NLP       | TextBlob   |
| ML        | scikit-learn |
| UI        | Streamlit  |
| Data      | pandas, numpy |

### Ethical Safeguards
- No medical diagnosis is made
- Data submission is voluntary
- No personally identifiable information is required
- System facilitates human support — does not replace it
    """)
