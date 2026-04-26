import pickle
from preprocess import preprocess_text

def predict_sentiment(tweet_text):
    """
    Predict sentiment of a single tweet.
    Returns: sentiment label (Positive / Negative / Neutral / Irrelevant)
    """
    tfidf   = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    model   = pickle.load(open("models/ensemble_model.pkl", "rb"))

    clean   = preprocess_text(tweet_text)
    vector  = tfidf.transform([clean])
    result  = model.predict(vector)[0]
    proba   = model.predict_proba(vector)[0]
    classes = model.classes_

    print(f"\nTweet    : {tweet_text}")
    print(f"Sentiment: {result.upper()}")
    print("\nConfidence Scores:")
    for cls, prob in zip(classes, proba):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12}: {prob:.2%}  {bar}")

    return result


if __name__ == "__main__":
    # Test with sample tweets
    test_tweets = [
        "I absolutely love this new phone! Best purchase ever!",
        "This service is terrible, I want my money back!",
        "The weather today is okay I guess.",
        "Just posted a new video on my channel check it out"
    ]

    print("=" * 55)
    print("  REAL-TIME SENTIMENT ANALYSIS DEMO")
    print("=" * 55)

    for tweet in test_tweets:
        predict_sentiment(tweet)
        print("-" * 55)
