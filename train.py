import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)
from preprocess import preprocess_text

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_PATH  = "data/twitter_sentiment.csv"   # Update if your file name differs
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Expected columns: id, topic/category, sentiment, tweet
# Rename to standard names if needed
df.columns = [c.lower().strip() for c in df.columns]

# Keep only needed columns — adjust column names to match your CSV
text_col      = "tweet"        # column with tweet text
sentiment_col = "sentiment"    # column with labels

df = df[[text_col, sentiment_col]].dropna()
df.columns = ["text", "sentiment"]

print(f"Dataset loaded: {len(df)} tweets")
print(f"Class distribution:\n{df['sentiment'].value_counts()}\n")

# ── PREPROCESS ───────────────────────────────────────────────────────────
print("Preprocessing tweets (this may take a few minutes)...")
df["clean_text"] = df["text"].apply(preprocess_text)

# ── FEATURE EXTRACTION ───────────────────────────────────────────────────
print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_text"])
y = df["sentiment"]

# ── TRAIN/TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# ── MODELS ───────────────────────────────────────────────────────────────
lr  = LogisticRegression(max_iter=1000, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf)],
    voting='soft'
)

models = {
    "Logistic Regression": lr,
    "Random Forest"      : rf,
    "Ensemble Model"     : ensemble
}

results = {}

# ── TRAIN & EVALUATE ─────────────────────────────────────────────────────
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

    results[name] = {
        "Accuracy" : round(acc, 4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec, 4),
        "F1-Score" : round(f1, 4),
        "ROC-AUC"  : round(auc, 4)
    }

    print(f"\n{name} Results:")
    print(f"  Accuracy : {acc:.2%}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# ── SAVE BEST MODEL ───────────────────────────────────────────────────────
print("\nSaving models and vectorizer...")
pickle.dump(ensemble, open(f"{MODEL_DIR}/ensemble_model.pkl", "wb"))
pickle.dump(lr,       open(f"{MODEL_DIR}/logistic_regression.pkl", "wb"))
pickle.dump(rf,       open(f"{MODEL_DIR}/random_forest.pkl", "wb"))
pickle.dump(tfidf,    open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb"))

# ── RESULTS SUMMARY ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
results_df = pd.DataFrame(results).T
print(results_df.to_string())
results_df.to_csv("models/results_summary.csv")
print("\nAll models saved in /models folder.")
