import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Common chat acronyms
CHAT_WORDS = {
    "omg": "oh my god",
    "lol": "laughing out loud",
    "brb": "be right back",
    "ttyl": "talk to you later",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "ngl": "not gonna lie",
    "irl": "in real life",
    "idk": "i do not know",
    "btw": "by the way",
    "gr8": "great",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "thx": "thanks",
    "pls": "please",
    "dm": "direct message",
    "rt": "retweet"
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 5. Expand chat acronyms
    words = text.split()
    words = [CHAT_WORDS.get(word, word) for word in words]
    text = ' '.join(words)

    # 6. Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 7. Remove stopwords and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # 8. Normalize whitespace
    text = ' '.join(words).strip()

    return text


if __name__ == "__main__":
    # Test the preprocessor
    sample = "OMG I LOVE this product!! 😍 Check it out: https://example.com #amazing @user"
    print("Original :", sample)
    print("Processed:", preprocess_text(sample))
