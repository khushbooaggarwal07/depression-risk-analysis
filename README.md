# Depression Risk Analysis Using Music Listening Patterns

> A machine learning pipeline for mood-based clustering and depression screening from Spotify audio features.

---

## Overview

This project proposes a scalable, passive approach to early depression screening by analysing music listening behaviour. Using ~114,000 Spotify tracks, the pipeline clusters songs into four mood profiles and trains supervised classifiers to identify listening patterns associated with depression risk — without any clinical infrastructure or self-report data.

**Paper:** *Depression Risk Analysis Using Music Listening Patterns: A Machine Learning Approach to Mood-Based Clustering and Classification*  
**Affiliation:** Department of Computer Science and Engineering / IoT and Intelligent Systems, Manipal University Jaipur

---

## Dataset
- **Size:** ~114,000 tracks across 114 genres, 19 original features
- **Final feature matrix:** 6 clinically motivated audio features after preprocessing

| Feature       | Description                                      | Depression Relevance                        |
|---------------|--------------------------------------------------|---------------------------------------------|
| Danceability  | Rhythmic regularity and beat strength (0–1)      | Low values linked to anhedonia              |
| Energy        | Perceptual intensity (0–1)                       | Primary acoustic marker of depressive listening |
| Loudness      | Normalised volume (0–1)                          | Depressed listeners prefer quieter content  |
| Speechiness   | Spoken-word presence (0–1)                       | Discriminates spoken-word from music genres |
| Valence       | Emotional positivity (0–1)                       | Strongest correlate of sadness vs. happiness |
| Tempo         | Estimated BPM                                    | Slower tempos correlate with depressive states |

---

## Pipeline

```
Raw Spotify Dataset (114K tracks, 19 features)
        ↓
Preprocessing & Feature Engineering
        ↓
K-Means Clustering (k=4, validated via Elbow + Silhouette)
        ↓
Mood Labels: Chill / Energetic / Cheerful / Romantic
        ↓
PCA & t-SNE Visualisation
        ↓
Train/Test Split (67% / 33%)
        ↓
SVM | KNN | Random Forest | MLP
        ↓
Evaluation: Accuracy, Precision, Recall, F1
        ↓
Depression Risk Mood Profile
```

---

## Mood Clusters

| Cluster    | Label      | Energy | Valence | Tempo      | Depression Signal      |
|------------|------------|--------|---------|------------|------------------------|
| 0          | Chill      | ~0.35  | ~0.28   | ~96 BPM    | Primary risk indicator |
| 1          | Energetic  | ~0.82  | —       | ~138 BPM   | Low risk               |
| 2          | Cheerful   | —      | ~0.78   | —          | Low risk               |
| 3          | Romantic   | Moderate | ~0.71 | —          | Low risk               |

Optimal k=4 confirmed by elbow inflection and silhouette score ≈ 0.53.

---

## Results

| Model         | Accuracy | Macro F1 | Weighted F1 | Rank |
|---------------|----------|----------|-------------|------|
| SVM           | **0.97** | 0.94     | **0.97**    | 1    |
| Random Forest | 0.91     | 0.73     | 0.87        | 2    |
| KNN           | 0.88     | 0.88     | 0.88        | 3    |
| MLP           | 0.45     | 0.16     | 0.28        | 4    |

**Key findings:**
- SVM dominates due to K-Means' inherently linear boundaries matching a linear-kernel SVM.
- Random Forest fails on the minority Cheerful class (F1=0.00); remediation: SMOTE / class-weighted loss.
- KNN is the most balanced baseline — non-zero F1 across all classes.
- MLP collapses to majority class (Chill); requires hyperparameter tuning and class balancing.
- Most discriminative features: **Energy (0.31) > Tempo (0.22) > Valence (0.19)** (Random Forest importance).

---

## Depression Risk Stratification

| Risk Level   | Listening Pattern                                    |
|--------------|------------------------------------------------------|
| Low          | Cheerful or Romantic > 60% of tracks over 30 days   |
| Moderate     | Mixed listening with Chill > 40%                    |
| High         | Chill > 60% of tracks for 14 or more consecutive days |

> **Disclaimer:** This is an early-warning indicator only, not a clinical diagnosis. High-risk signals should prompt completion of PHQ-9 or referral to a care coordinator.

---

## Repository Structure

```
├── dataset2 (1).csv                # Spotify tracks dataset (~114K tracks, 19 features)
├── depression_risk_analysis.ipynb  # Full pipeline: preprocessing → clustering → classification → evaluation
└── README.md
```

---

## Installation

```bash
git clone https://github.com/khushbooaggarwal07/depression-risk-analysis.git
cd depression-risk-analysis
pip install pandas numpy scikit-learn matplotlib seaborn
```

Then open `depression_risk_analysis.ipynb` in Jupyter or Google Colab and run all cells.

---


---

## Future Work

- **Temporal modelling** — LSTM/Transformer over 14-day listening windows
- **Multi-modal fusion** — Lyrical sentiment (BERT), wearable data (HRV, sleep)
- **Personalised models** — Transfer learning fine-tuned per user
- **Clinical validation** — Prospective PHQ-9 + Spotify data collection study
- **Explainability** — Per-user SHAP values for regulatory/clinical use
- **Real-time deployment** — Spotify Web API integration with a clinical dashboard

---
