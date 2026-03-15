# AI-Based Peer Support Recommendation System
### A Cognitive Analytics Framework for Intelligent Peer Matching

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview
An explainable AI system that detects user support needs using behavioral signals, classifies the support type (Emotional / Academic / Motivational), and recommends the top-K most compatible peers using multi-factor similarity matching — with transparent reasoning for every recommendation.

---

## System Architecture

```
User Input  →  Feature Engineering  →  Cognitive Scoring
            →  Peer Matching         →  Explainable Output
```

**5 Layers:**
1. User Input Layer — mood, engagement, reflection text, domain, skills, availability, experience
2. Feature Engineering Layer — NLP pipeline (TextBlob sentiment, negative word frequency, engagement delta, skill vectors)
3. Cognitive Need Scoring Engine — rule-based, transparent scoring
4. Peer Matching Engine — `MatchScore = w₁·D + w₂·E + w₃·C + w₄·A`
5. Recommendation Output Layer — Top-K peers with per-factor explanations

---

## Project Structure

```
peer_support_system/
├── app.py                        # Streamlit UI (main entry point)
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator (520 users)
│   └── users.csv                 # Generated dataset
├── utils/
│   └── feature_engineering.py   # NLP + feature transformation pipeline
├── models/
│   ├── cognitive_scorer.py       # Rule-based need scoring engine
│   ├── peer_matcher.py           # Similarity-based matching engine
│   └── recommendation_engine.py # Top-level facade / pipeline
├── evaluation/
│   └── evaluator.py              # Evaluation metrics & baseline comparison
└── outputs/                      # Saved results / reports
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Run evaluation only
```bash
python evaluation/evaluator.py
```

---

## Matching Formula

```
MatchScore = w₁ × D  +  w₂ × E  +  w₃ × C  +  w₄ × A

D = Domain similarity       (1.0 exact, 0.5 related cluster, 0.0 unrelated)
E = Experience compatibility (1.0 one level above — ideal mentor)
C = Skill cosine similarity  (cosine of multi-hot skill vectors)
A = Availability overlap     (Jaccard similarity of time-slot sets)
```

Default weights: `w₁=0.30, w₂=0.25, w₃=0.25, w₄=0.20` (adjustable in sidebar)

---

## Evaluation Results (n=200 queries, K=3)

| Metric                 | Value  |
|------------------------|--------|
| Precision@K (System)   | 0.9967 |
| Precision@K (Random)   | 0.6833 |
| Matching Accuracy      | 1.0000 |
| User Satisfaction      | 0.7138 |
| False Positive Rate    | 0.0033 |
| Improvement vs Random  | +0.3133|

---

## Cognitive Need Scoring Rules

| Score       | Key Rules                                                    |
|-------------|--------------------------------------------------------------|
| Emotional   | Low mood (≤4) + negative sentiment + distress keywords       |
| Academic    | Low engagement (≤40%) + engagement drop + difficulty keywords|
| Motivational| Neutral mood (4-7) + moderate engagement + flat tone         |

---

## Tech Stack

| Component   | Technology               |
|-------------|--------------------------|
| Backend     | Python 3.9+              |
| NLP         | TextBlob                 |
| ML/Similarity | scikit-learn (cosine)  |
| Data        | pandas, numpy            |
| UI          | Streamlit                |
| Dataset     | Synthetic (520 users)    |

---

## Ethical Statement
This system is designed to **facilitate** human peer support, not replace professional mental health care. No medical diagnosis is made. All data submission is voluntary. No personally identifiable information is required.

---

## License
MIT
