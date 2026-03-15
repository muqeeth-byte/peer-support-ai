"""
Streamlit UI — AI-Based Peer Support Recommendation System
"""
import os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import streamlit as st

st.set_page_config(page_title="PeerMatch AI", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1f4e79;text-align:center;padding:1rem 0}
.sub-header{font-size:1rem;color:#555;text-align:center;margin-bottom:2rem}
.score-card{background:#f8f9fa;border-radius:10px;padding:1rem;border-left:5px solid #1f4e79;margin:.5rem 0}
.peer-card{background:white;border-radius:10px;padding:1.2rem;border:1px solid #e0e0e0;
           box-shadow:0 2px 4px rgba(0,0,0,.08);margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

DOMAIN_LIST = ["Computer Science","Mathematics","Psychology","Engineering",
               "Biology","Business","Literature","Physics","Data Science","Education"]
SKILL_LIST  = ["Python","Statistics","Writing","Research","Communication",
               "Problem Solving","Data Analysis","Leadership","Critical Thinking",
               "Machine Learning","Project Management","Presentation",
               "Programming","Teamwork","Time Management"]
AVAIL_LIST  = ["Morning","Afternoon","Evening","Night","Weekends","Flexible"]
EXP_LIST    = ["Beginner","Intermediate","Advanced","Expert"]
COL_MAP     = {
    "Mood":"mood","Engagement":"engagement","EngagementDelta":"engagement_delta",
    "sentiment_compound":"sentiment_compound","negative_word_freq":"negative_word_freq",
    "academic_signal":"academic_signal","withdrawal_signal":"withdrawal_signal",
    "motivation_signal":"motivation_signal",
}

# ── Load peer pool (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_peer_pool():
    enriched = os.path.join(BASE_DIR,"data","peer_support_enriched.csv")
    raw      = os.path.join(BASE_DIR,"data","peer_support_dataset.csv")
    if os.path.exists(enriched):
        return pd.read_csv(enriched)
    from data.generate_dataset import generate_dataset
    from utils.nlp_features import extract_features_batch
    from models.cognitive_scorer import score_dataframe
    df = generate_dataset(n=500, save_path=raw)
    df = extract_features_batch(df)
    df = score_dataframe(df)
    df.to_csv(enriched, index=False)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤝 PeerMatch AI")
    st.markdown("""
**AI-Based Peer Support System**
- 🧠 Detects support needs via behavioral signals  
- 💬 NLP analysis of text reflections  
- 🔗 Weighted similarity peer matching  
- 📊 Fully explainable recommendations
""")
    st.divider()
    st.markdown("### ⚙️ Matching Weights")
    w_d = st.slider("Domain Similarity",    0.0, 1.0, 0.30, 0.05)
    w_e = st.slider("Experience Match",     0.0, 1.0, 0.25, 0.05)
    w_c = st.slider("Communication",        0.0, 1.0, 0.25, 0.05)
    w_a = st.slider("Availability Overlap", 0.0, 1.0, 0.20, 0.05)
    total_w = w_d + w_e + w_c + w_a
    st.caption(f"Total: {total_w:.2f}  (weights auto-normalised)")
    st.divider()
    top_k = st.selectbox("Recommendations to show", [1,2,3,5], index=2)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🤝 PeerMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cognitive Analytics Framework for Intelligent Peer Matching</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 Check-In & Recommendations","📊 Dashboard","📋 Methodology"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHECKIN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Step 1 — Tell us about yourself today")
    c1, c2 = st.columns(2)
    with c1:
        mood       = st.slider("Mood Score (1 = very low, 10 = excellent)", 1.0, 10.0, 5.0, 0.5)
        engagement = st.slider("Current Engagement Level", 0.0, 1.0, 0.5, 0.05)
        eng_delta  = st.slider("Engagement Change (vs last period)", -0.5, 0.5, 0.0, 0.05,
                               help="Negative = dropped, Positive = improved")
        domain     = st.selectbox("Academic Domain", DOMAIN_LIST)
    with c2:
        experience   = st.selectbox("Experience Level", EXP_LIST)
        availability = st.multiselect("Your Availability", AVAIL_LIST, default=["Evening"])
        skills       = st.multiselect("Key Skills", SKILL_LIST, default=["Communication"])

    st.subheader("Step 2 — Optional: Share a reflection")
    reflection = st.text_area("How are you feeling about your academic journey today?",
                              placeholder="e.g., I feel overwhelmed with coursework and can't focus...",
                              height=90)

    st.divider()
    if st.button("🔍 Find My Peer Matches", type="primary", use_container_width=True):
        with st.spinner("Analysing your profile and finding best matches..."):
            from utils.nlp_features import extract_features
            from models.cognitive_scorer import compute_cognitive_scores, generate_need_explanation
            from models.peer_matcher import recommend_peers

            nlp = extract_features(reflection)

            cog_kwargs = {
                "mood": mood, "engagement": engagement, "engagement_delta": eng_delta,
                **{k: nlp.get(k, 0.0) for k in
                   ["sentiment_compound","negative_word_freq","academic_signal",
                    "withdrawal_signal","motivation_signal"]},
            }
            result       = compute_cognitive_scores(**cog_kwargs)
            need_reasons = generate_need_explanation(result)

            user_series = pd.Series({
                "UserID":       "You",
                "Mood":         mood,
                "Engagement":   engagement,
                "EngagementDelta": eng_delta,
                "Domain":       domain,
                "Skills":       ", ".join(skills) if skills else "Communication",
                "Experience":   experience,
                "Availability": ", ".join(availability) if availability else "Evening",
                **nlp,
            })

            peer_pool = load_peer_pool()
            norm = total_w if total_w > 0 else 1.0
            weights = {"domain": w_d/norm, "experience": w_e/norm,
                       "communication": w_c/norm, "availability": w_a/norm}
            recs = recommend_peers(user_series, peer_pool, result.support_type,
                                   top_k=top_k, weights=weights)

        st.success("✅ Analysis complete!")

        # — Support need card ———————————————————————————————————————
        st.subheader("🧠 Support Need Detection")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Emotional Score",    f"{result.emotional_score:.1f}")
        mc2.metric("Academic Score",     f"{result.academic_score:.1f}")
        mc3.metric("Motivational Score", f"{result.motivational_score:.1f}")

        badge_colors = {"Emotional":"#ffe0e0","Academic":"#e0f0ff",
                        "Motivational":"#e8f5e9","None":"#f3f3f3"}
        text_colors  = {"Emotional":"#c0392b","Academic":"#1a5276",
                        "Motivational":"#1b5e20","None":"#555"}
        bc = badge_colors.get(result.support_type, "#f3f3f3")
        tc = text_colors.get(result.support_type, "#555")
        st.markdown(f"""
        <div class="score-card">
          <strong>Detected Support Need:</strong>
          <span style="background:{bc};color:{tc};padding:3px 12px;border-radius:20px;
                       font-weight:600;margin-left:10px;">{result.support_type}</span>
          <span style="color:#888;margin-left:12px;">Confidence: {result.confidence:.0%}</span>
        </div>""", unsafe_allow_html=True)

        if need_reasons:
            st.markdown("**Why support is suggested:**")
            for r in need_reasons:
                st.markdown(f"• {r}")

        st.divider()

        # — Recommendations ————————————————————————————————————————
        if result.support_type == "None":
            st.info("🎉 You appear to be doing well — no urgent peer support needed right now. "
                    "Feel free to connect with peers for collaboration or learning!")
        else:
            st.subheader(f"🎯 Top {len(recs)} Recommended Peers")
            for i, rec in enumerate(recs, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="peer-card">
                      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                        <div style="background:#1f4e79;color:white;border-radius:50%;
                                    width:30px;height:30px;display:flex;align-items:center;
                                    justify-content:center;font-weight:700">{i}</div>
                        <h4 style="margin:0;color:#1f4e79">{rec['peer_id']}</h4>
                        <span style="margin-left:auto;background:#e8f4fd;color:#1a5276;
                                     padding:3px 12px;border-radius:20px;font-weight:600">
                          Match: {rec['match_score']:.3f}
                        </span>
                      </div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
                        <div><b>Domain:</b> {rec['peer_domain']}</div>
                        <div><b>Experience:</b> {rec['peer_experience']}</div>
                        <div><b>Availability:</b> {rec['peer_availability']}</div>
                        <div><b>Skills:</b> {str(rec['peer_skills'])[:45]}...</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    fc1,fc2,fc3,fc4 = st.columns(4)
                    fc1.metric("Domain",        f"{rec['factors']['domain_score']:.2f}")
                    fc2.metric("Experience",    f"{rec['factors']['experience_score']:.2f}")
                    fc3.metric("Communication", f"{rec['factors']['communication_score']:.2f}")
                    fc4.metric("Availability",  f"{rec['factors']['availability_score']:.2f}")

                    with st.expander(f"Why {rec['peer_id']}?"):
                        for r in rec["explanation"]:
                            st.markdown(f"→ {r}")
                    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Analytics Dashboard")
    if st.button("Load Analytics", key="load_analytics"):
        df = load_peer_pool()

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Users", len(df))
        m2.metric("Emotional",    len(df[df["PredictedSupport"]=="Emotional"]))
        m3.metric("Academic",     len(df[df["PredictedSupport"]=="Academic"]))
        m4.metric("Motivational", len(df[df["PredictedSupport"]=="Motivational"]))

        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Support Type Distribution")
            st.bar_chart(df["PredictedSupport"].value_counts())
        with c2:
            st.subheader("Experience Level Distribution")
            st.bar_chart(df["Experience"].value_counts())

        c3,c4 = st.columns(2)
        with c3:
            st.subheader("Mood Score Distribution")
            bins = pd.cut(df["Mood"], bins=[0,2,4,6,8,10],
                          labels=["1–2","3–4","5–6","7–8","9–10"])
            st.bar_chart(bins.value_counts().sort_index())
        with c4:
            st.subheader("Domain Distribution")
            st.bar_chart(df["Domain"].value_counts())

        st.subheader("Sample Records")
        cols = ["UserID","Mood","Engagement","Domain","Experience","PredictedSupport","SupportConfidence"]
        st.dataframe(df[[c for c in cols if c in df.columns]].head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 System Methodology")
    st.markdown("""
### 5-Layer Architecture

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | User Input | Mood, engagement, text, domain, skills, availability |
| 2 | Feature Engineering | VADER/heuristic sentiment, keyword signals, encoding |
| 3 | Cognitive Scoring | Rule-based Emotional / Academic / Motivational scores |
| 4 | Peer Matching | MatchScore = w₁D + w₂E + w₃C + w₄A |
| 5 | Output | Top-K peers with transparent explanation chain |

### Matching Formula
```
MatchScore = w₁·D + w₂·E + w₃·C + w₄·A

D = Domain Similarity    (1.0 same / 0.5 related / 0.0 different)
E = Experience Match     (higher peer exp → higher score)
C = Communication Compat (cosine similarity on skill vectors)
A = Availability Overlap (Jaccard similarity on time slots)
```

### Cognitive Scoring Rules
| Need | Signals |
|------|---------|
| Emotional | Mood ≤ 4 + negative sentiment + withdrawal keywords |
| Academic  | Engagement ≤ 0.4 + engagement drop + academic keywords |
| Motivational | Mood 4–7 + engagement 0.4–0.6 + motivational keywords |

### Evaluation Metrics
- **Precision@K** — fraction of top-K recommendations that are relevant
- **Matching Accuracy** — agreement between predicted and true support type
- **False Positive Rate** — users incorrectly flagged when doing well
- **User Satisfaction** — fraction of recommendations with match score ≥ 0.5

### Ethical Principles
✅ Voluntary input only &nbsp;&nbsp; ✅ No medical diagnosis &nbsp;&nbsp;
✅ Fully explainable &nbsp;&nbsp; ✅ User-controlled weights &nbsp;&nbsp;
✅ Facilitates human support — does not replace it
""")
