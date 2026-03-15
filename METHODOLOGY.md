# Research Methodology Section
## AI-Based Peer Support Recommendation System: A Cognitive Analytics Framework for Intelligent Peer Matching

---

### 3. METHODOLOGY

#### 3.1 System Overview

This study proposes a modular, five-layer AI framework for intelligent peer support
recommendation. Unlike content-based or collaborative filtering systems designed for
product or media recommendation, the proposed system is purpose-built for structured
human support ecosystems. The framework employs cognitive analytics, natural language
processing (NLP), and weighted similarity modelling to detect support needs and recommend
suitable peers, while maintaining full transparency through explainable AI (XAI) principles.

---

#### 3.2 Dataset

A synthetic dataset of 500 user profiles was generated to simulate realistic behavioral
patterns in academic and professional peer support environments. Each user profile includes:

- **Mood Score** (1–10): Self-reported wellbeing rating
- **Engagement Score** (0–1): Participation level in academic activities
- **Engagement Delta**: Change in engagement relative to the prior session
- **Text Reflection**: Free-form written check-in (optional)
- **Academic Domain**: Field of study or professional domain
- **Skills**: A set of up to five competencies from a predefined pool of 15
- **Experience Level**: Categorical label (Beginner, Intermediate, Advanced, Expert)
- **Availability**: Time slots available for peer interaction

Profiles were generated across four behavioral archetypes — Emotional (25%), Academic
(25%), Motivational (25%), and Neutral/Balanced (25%) — to enable controlled
evaluation against ground-truth support labels.

---

#### 3.3 Feature Engineering

Raw inputs are transformed into quantifiable behavioral features through two pipelines:

**3.3.1 NLP Feature Extraction**

Text reflections are processed to extract the following signals:

| Feature | Method |
|---------|--------|
| Sentiment Compound | VADER lexicon-based polarity scoring |
| Negative Sentiment Score | VADER negative fraction |
| Negative Word Frequency | Keyword matching against a distress lexicon (28 terms) |
| Academic Signal | Keyword matching against an academic distress lexicon (20 terms) |
| Withdrawal Signal | Keyword matching against a social withdrawal lexicon (11 terms) |
| Motivation Signal | Keyword matching against a motivational deficit lexicon (13 terms) |
| Text Length (normalised) | Token count normalised to [0, 1] over 30-word baseline |

VADER (Valence Aware Dictionary and sEntiment Reasoner) was selected for its
validated performance on short informal text and its ability to score sentences without
requiring model training, which preserves explainability.

**3.3.2 Structural Feature Encoding**

- **Domain**: Encoded as an integer index across 10 academic domains
- **Skills**: Encoded as a 15-dimensional binary vector (multi-hot)
- **Availability**: Encoded as a 6-dimensional binary vector
- **Experience**: Mapped to an ordinal integer (0–3)

---

#### 3.4 Cognitive Need Scoring Engine

The core classification module employs a rule-based scoring model that computes three
independent support-need scores from behavioral indicators:

**Emotional Support Score (E)**

Triggered by the following rule set:

- Rule E1: Mood ≤ 4.0 → +3.0 points
- Rule E2: VADER compound < −0.10 → +2.0 points
- Rule E3: Negative keyword count ≥ 3 (normalised) → +2.0 points; ≥ 1 → +1.0
- Rule E4: Withdrawal keyword presence → +1.5 points
- Rule E5: Engagement_delta < −0.10 AND Engagement < 0.35 → +1.5 points

**Academic Support Score (A)**

- Rule A1: Engagement ≤ 0.40 → +3.0 points
- Rule A2: Engagement_delta < −0.10 → +2.0 points
- Rule A3: Academic signal keywords detected → +2.0 points
- Rule A4: Mood ≥ 5 AND Engagement ≤ 0.40 → +1.0 points (mood-engagement mismatch)

**Motivational Support Score (M)**

- Rule M1: 4.0 < Mood ≤ 7.0 → +2.0 points
- Rule M2: 0.40 < Engagement ≤ 0.60 → +2.0 points
- Rule M3: −0.10 ≤ Sentiment_compound ≤ 0.10 → +1.0 points
- Rule M4: Motivation signal keywords detected → +1.5 points
- Rule M5: −0.10 ≤ Engagement_delta < 0 → +1.0 points

**Classification**: The category with the highest score is selected as the primary
support type. If all scores fall below a minimum threshold of 2.0, the user is
classified as not requiring active support ("None").

**Confidence** is computed as:

    Confidence = min(max_score / 10.0, 1.0)

This design avoids opaque probabilistic predictions and ensures every classification
decision can be directly explained by the active rule set.

---

#### 3.5 Peer Matching Engine

Following support type classification, the system performs similarity-based peer
matching using a weighted linear composite score:

    MatchScore = w₁·D + w₂·E + w₃·C + w₄·A

Where each factor is computed as follows:

**D — Domain Similarity**

    D = 1.0  if user_domain == peer_domain
    D = 0.5  if domains belong to the same related cluster
    D = 0.0  otherwise

Domain clusters are predefined (e.g., {Computer Science, Data Science, Mathematics,
Engineering, Physics}).

**E — Experience Match**

Computed as a function of the ordinal difference between peer and user experience:

    diff = peer_exp_ordinal − user_exp_ordinal
    E = 1.0 (diff=0), 0.75 (diff=1), 0.50 (diff=2), 0.30 (diff≥3)
    E = max(0.0, 0.4 + diff·0.15)  for diff < 0

Higher-experience peers are preferred, reflecting the mentorship model of peer support.

**C — Communication Compatibility**

Computed as the cosine similarity between the two multi-hot skill vectors:

    C = cos(vec_user_skills, vec_peer_skills)

**A — Availability Overlap**

Computed as the Jaccard similarity of the two availability slot sets:

    A = |slots_user ∩ slots_peer| / |slots_user ∪ slots_peer|

A value of 1.0 is assigned if either party marks "Flexible".

**Default weights**: w₁=0.30, w₂=0.25, w₃=0.25, w₄=0.20. These are user-adjustable
via the system interface to accommodate different deployment contexts.

**Top-K Selection**: Peers are ranked by MatchScore in descending order and the
top K (default K=3) are returned with a full factor breakdown.

---

#### 3.6 Explainability Mechanism

Each recommendation is accompanied by a structured explanation chain that identifies:

1. **Why support is needed**: the specific rules activated in the Cognitive Scoring Engine
2. **Why each peer was selected**: the factor scores (D, E, C, A) and their verbal interpretation
3. **Confidence levels**: for both support classification and match quality

This design satisfies the XAI requirement of *transparency by design* (Arrieta et al., 2020),
ensuring that neither the system nor the user operates within a black-box context.

---

#### 3.7 Evaluation

System performance was assessed using four complementary metrics:

| Metric | Definition |
|--------|-----------|
| **Matching Accuracy** | Fraction of users where PredictedSupport == TrueSupport (ground truth) |
| **Precision@K** | Fraction of top-K recommended peers that satisfy the relevance criteria |
| **False Positive Rate** | Fraction of "None" users incorrectly flagged as needing support |
| **User Satisfaction (proxy)** | Fraction of recommendations with MatchScore ≥ 0.50 |

**Relevance** for Precision@K is defined heuristically: a peer is relevant if they
share the same or a related academic domain AND have an experience level ≥ the user.

All metrics are compared against a **random matching baseline** (uniformly random peer
selection, averaged over 100 trials per user) to quantify the improvement attributable
to the intelligent matching algorithm.

**Experimental results** (n=500 users, K=3, sample=100 for recommendation evaluation):

| Metric | System | Random Baseline |
|--------|--------|-----------------|
| Matching Accuracy | 75.4% | N/A |
| False Positive Rate | 40.9% | N/A |
| Precision@K | 1.0000 | 0.2371 |
| Improvement | +0.7629 | — |
| User Satisfaction | 100% | — |
| Avg Match Score | 0.849 | — |

The system achieves a **+76.3 percentage point improvement in Precision@K** over the
random baseline, demonstrating that the intelligent matching algorithm produces
substantially more relevant recommendations than chance selection.

---

#### 3.8 Ethical Considerations

The system is designed with the following ethical safeguards:

- **Voluntary participation**: All inputs are user-submitted; no behavioural data is
  inferred without explicit consent.
- **No clinical diagnosis**: The system detects behavioural patterns only; it makes no
  medical or psychological diagnoses.
- **Privacy preservation**: No personally identifiable information is required.
- **User control**: Matching weights are user-adjustable, preserving agency over
  the recommendation process.
- **Human-in-the-loop**: The system facilitates peer connections; all support
  interactions occur between humans without AI mediation.

---

#### References

- Arrieta, A.B. et al. (2020). Explainable Artificial Intelligence (XAI): Concepts,
  taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82–115.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. *ICWSM*.
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Salton, G. & McGill, M.J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
