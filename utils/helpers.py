import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Ensure NLTK stopwords are available
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Cleans text by lowercasing, removing non-alphabetic chars, and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return ' '.join(words)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers more nuanced features from the review text.
    This version avoids using direct length/word counts as primary features.
    """
    features = pd.DataFrame()
    
    # Feature 1: Ratio of uppercase words to total words
    total_words = df['review_text'].apply(lambda x: len(str(x).split()))
    upper_case_words = df['review_text'].apply(lambda x: len([word for word in str(x).split() if word.isupper()]))
    features['upper_case_ratio'] = upper_case_words / total_words.replace(0, 1)

    # Feature 2: Count of exclamation marks
    features['exclamation_count'] = df['review_text'].str.count('!')

    # Feature 3: Average word length in the review
    # Short average word length can indicate simple/spammy text
    words = df['review_text'].apply(lambda x: str(x).split())
    word_lengths = words.apply(lambda x: [len(word) for word in x])
    features['avg_word_length'] = word_lengths.apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

    return features.fillna(0)
