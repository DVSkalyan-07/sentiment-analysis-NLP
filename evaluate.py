import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import label_binarize
import os

os.makedirs("outputs", exist_ok=True)

def plot_confusion_matrix(y_test, y_pred, model_name, classes):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix – {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    fname = f"outputs/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_roc_curve(y_test, y_prob, model_name, classes):
    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
                 label=f'{cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve – {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    fname = f"outputs/roc_curve_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def evaluate(X_test, y_test):
    # Load models
    models = {
        "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
        "Random Forest"      : pickle.load(open("models/random_forest.pkl", "rb")),
        "Ensemble Model"     : pickle.load(open("models/ensemble_model.pkl", "rb"))
    }

    classes = sorted(y_test.unique().tolist())

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))
        plot_confusion_matrix(y_test, y_pred, name, classes)
        plot_roc_curve(y_test, y_prob, name, classes)

    print("\nAll evaluation plots saved in /outputs folder.")


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from preprocess import preprocess_text

    print("Loading data for evaluation...")
    df = pd.read_csv("data/twitter_sentiment.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["tweet", "sentiment"]].dropna()
    df.columns = ["text", "sentiment"]
    df["clean_text"] = df["text"].apply(preprocess_text)

    tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    X = tfidf.transform(df["clean_text"])
    y = df["sentiment"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    evaluate(X_test, y_test)
