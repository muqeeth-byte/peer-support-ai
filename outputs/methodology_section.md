# Methodology — AI-Based Peer Support Recommendation System

## 3. Methodology

### 3.1 System Overview

This study proposes a five-layer modular pipeline that transforms raw behavioral check-in data into structured, explainable peer support recommendations. The architecture separates concerns cleanly: input collection, feature transformation, cognitive scoring, peer ranking, and output explanation are handled by independent, testable components.

---

### 3.2 Dataset

A synthetic dataset of 520 user records was generated to simulate diverse academic and socio-behavioral profiles. Each record includes: a self-reported mood score (1–10), a session engagement score (0–1), a previous-session engagement score for delta computation, a free-text reflection, an academic domain, a pipe-delimited skill list, availability time-slots, and an experience level label.

User profiles were assigned to three behavioral archetypes — positive (35%), neutral (30%), and negative (35%) — with reflection texts, mood, and engagement values drawn from archetype-specific distributions to ensure realistic diversity.

---

### 3.3 Feature Engineering

Raw inputs are transformed into structured numeric features via a multi-step NLP pipeline:

- **Sentiment Polarity**: TextBlob's pattern-based sentiment analyser assigns a polarity score in [−1, +1] to each reflection text.
- **Negative Word Frequency**: A curated lexicon of 25 psychological distress keywords counts high-signal terms (e.g., overwhelmed, hopeless, struggling).
- **Engagement Delta**: The signed difference between current and previous engagement scores captures temporal participation trends.
- **Domain Encoding**: Academic domains are label-encoded and grouped into five relatedness clusters for partial-credit matching.
- **Skill Vectors**: Multi-hot binary vectors of length 15 represent user skill profiles.
- **Availability Vectors**: Multi-hot binary vectors of length 7 encode time-slot availability.
- **Experience Encoding**: Ordinal encoding assigns integer values 1–4 to Beginner/Intermediate/Advanced/Expert levels.

---

### 3.4 Cognitive Need Scoring Engine

A rule-based explainable scoring model assigns independent scores for three support categories without relying on opaque predictive models.

**Emotional Support Score:**
- +3.0 if mood <= 4 (strong distress signal)
- +2.0 if sentiment polarity < -0.10 (negative text tone)
- +2.0 if negative keyword count >= 3 (multiple distress terms)
- +1.5 if engagement delta < -0.10 AND current engagement < 0.35 (withdrawal pattern)

**Academic Support Score:**
- +3.0 if engagement <= 0.40 (low session participation)
- +2.0 if engagement delta < -0.10 (declining participation trend)
- +1.5 if negative keyword count >= 2 (academic difficulty vocabulary)
- +1.0 if mood >= 5 AND engagement <= 0.40 (capability-participation mismatch)

**Motivational Support Score:**
- +2.0 if 4 < mood <= 7 (neutral mood range)
- +2.0 if 0.40 < engagement <= 0.60 (moderate participation plateau)
- +1.0 if -0.10 <= sentiment <= 0.10 (flat reflective tone)
- +1.0 if -0.10 <= engagement delta < 0 (mild downward trend)

The support category with the highest score is selected as the primary recommendation type.

---

### 3.5 Peer Matching Engine

For a query user q, compatibility with each candidate peer p is:

    MatchScore(q, p) = w1*D + w2*E + w3*C + w4*A

Where:
- **D — Domain Similarity**: 1.0 identical; 0.5 same cluster; 0.0 unrelated.
- **E — Experience Compatibility**: 1.0 peer one level above (ideal mentor); 0.9 same level; 0.7 two levels above.
- **C — Skill Compatibility**: Cosine similarity of multi-hot skill vectors.
- **A — Availability Overlap**: Jaccard similarity of time-slot sets.

Default weights: w1=0.30, w2=0.25, w3=0.25, w4=0.20 (adjustable).

---

### 3.6 Evaluation Results

| Metric                    | System  | Random Baseline |
|---------------------------|---------|-----------------|
| Precision@3               | 0.9967  | 0.6833          |
| Matching Accuracy         | 1.0000  | —               |
| User Satisfaction (proxy) | 0.7138  | —               |
| False Positive Rate       | 0.0033  | —               |

System achieved +0.3133 improvement in Precision@K over random baseline.
