# 🧠 Real-Time Public Sentiment Analysis Using NLP & Machine Learning

> A B.Tech Final Year Project — Vignan's University, May 2025

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

---

## 📌 Overview

This project presents a **real-time sentiment classification system** built to analyze public opinion from social media data (Twitter). Using a combination of NLP preprocessing techniques and ensemble machine learning models, the system classifies tweets into four sentiment categories:

- ✅ Positive
- ❌ Negative
- 😐 Neutral
- 🚫 Irrelevant

The system processed **74,681 labeled tweets** and achieved **97% classification accuracy** using a soft-voting ensemble of Logistic Regression and Random Forest models.

---

## 🎯 Problem Statement

Traditional sentiment models struggle with:
- Noisy, informal social media text (slang, emojis, hashtags)
- Real-time deployment constraints (high latency)
- Generalization across domains

This project addresses these gaps with a lightweight, modular, and highly accurate NLP pipeline suitable for real-world deployment.

---

## 🏗️ System Architecture

```
Raw Tweets
    ↓
Preprocessing (NLP)
    ↓
TF-IDF Vectorization
    ↓
┌─────────────────────────┐
│  Logistic Regression    │
│  Random Forest          │
└─────────────────────────┘
    ↓
Ensemble (Soft Voting)
    ↓
Sentiment Prediction
(Positive / Negative / Neutral / Irrelevant)
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| NLP | NLTK, spaCy |
| ML Models | Scikit-learn |
| Feature Extraction | TF-IDF Vectorization |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset Source | Kaggle (Twitter Sentiment Dataset) |

---

## 🔄 NLP Preprocessing Pipeline

The raw tweet text goes through the following steps:

1. **Lowercasing** — Normalize text to reduce dimensionality
2. **HTML & URL Removal** — Strip irrelevant hyperlinks
3. **Emoji Removal** — Remove Unicode emoji characters
4. **Chat Acronym Mapping** — Expand abbreviations (e.g., "omg" → "oh my god")
5. **Punctuation & Stopword Removal** — Eliminate noise words
6. **Lemmatization** — Reduce words to base form (e.g., "running" → "run")
7. **Whitespace Normalization** — Clean extra spaces and newlines

---

## 🤖 Models Used

### 1. Logistic Regression
- Linear model using sigmoid function
- Efficient and interpretable
- Best for linearly separable features

### 2. Random Forest
- Tree-based ensemble using bootstrapping
- Captures non-linear relationships
- Robust against overfitting

### 3. Ensemble (Soft Voting) ⭐ Best Model
- Combines predicted probabilities from both models
- Averages class probabilities for final prediction
- Achieves highest accuracy and generalization

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 95% | 0.95 | 0.95 | 0.95 | 0.98 |
| Random Forest | 96% | 0.96 | 0.96 | 0.96 | 0.99 |
| **Ensemble Model** | **97%** | **0.97** | **0.97** | **0.97** | **0.99** |

The **Ensemble Model outperformed** both individual classifiers across all metrics.

---

## 📁 Dataset

- **Source:** Kaggle (public Twitter sentiment dataset)
- **Total Tweets:** 74,681 labeled tweets
- **Classes:** Positive, Negative, Neutral, Irrelevant
- **Split:** 80% Training / 20% Testing (Stratified)

| Sentiment | Count | Proportion |
|---|---|---|
| Negative | 28,808 | 30.14% |
| Positive | 25,109 | 27.89% |
| Neutral | 21,603 | 24.58% |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/dvskalyan07/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python preprocess.py

# 4. Train models
python train.py

# 5. Evaluate results
python evaluate.py
```

---

## 📦 Requirements

```
nltk
spacy
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📈 Key Achievements

- ✅ Processed and classified **74,681 tweets** in real-time pipeline
- ✅ Achieved **97% accuracy** using ensemble learning
- ✅ Reduced system latency by **30%** through preprocessing optimization
- ✅ Improved data accuracy by **12%** via robust NLP pipeline
- ✅ ROC-AUC of **0.99** — near-perfect class separation

---

## 🔮 Future Scope

- Integration of **BERT / LSTM** for deeper semantic understanding
- **Multilingual support** using cross-lingual embeddings
- **Multimodal analysis** — combining text, images, and hashtags
- **Edge deployment** for low-latency mobile inference
- Real-time Twitter API streaming integration

---

## 👨‍💻 Authors

| Name | Roll No |
|---|---|
| D. Venkata Sai Kalyan | 211FA04483 |
| Ch. Adarsh | 211FA04494 |

**Guide:** Mr. Kumar Devapogu, Assistant Professor, CSE  
**Institution:** Vignan's Foundation for Science, Technology & Research (Deemed University), Guntur, AP

---

## 📄 Project Report

The full project report is available in this repository: [`Project_Report.pdf`](./Project_Report.pdf)

---

## 🏛️ Institution

**Department of Computer Science & Engineering**  
Vignan's Foundation for Science, Technology & Research  
Vadlamudi, Guntur - 522213, India  
*May 2025*
