import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

text = "Natural Language Processing is AMAZING! It helps computers understand humans."

print("\nOriginal text:", text)

# 1. Tokenization
tokens = word_tokenize(text)
print("\nTokens:", tokens)

# 2. Lowercasing
tokens_lower = [t.lower() for t in tokens]

# 3. Remove punctuation
tokens_clean = [ t for t in tokens_lower if t not in string.punctuation]

# 4. Stopword removal
stop_words = set(stopwords.words("english"))
tokens_no_stop = [t for t in tokens_clean if t not in stop_words]
print("\nWithout stopwords:", tokens_no_stop)

print("\nWithout stopwords:", tokens_no_stop)

# 5. Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in tokens_no_stop]
print("\nStemmed tokens:", stemmed)

print("\nStemmed tokens:", stemmed)

# 6. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in tokens_no_stop]
print("\nLemmatized tokens:", lemmatized)

# Sentiment analysis

positive_words = ["good", "great", "excellent", "love", "amazing", "happy"]
negative_words = ["bad", "terrible", "sad", "hate", "awful", "poor"]

def simple_sentiment(text):
    tokens = word_tokenize(text.lower())
    score = 0

    for w in tokens:
        if w in positive_words:
            score += 1
        if w in negative_words:
            score -= 1
    
    if score > 0: return "Positive"
    if score < 0: return "Negative"
    return "Neutral"

print(simple_sentiment("I love AI, it is amazing!"))
print(simple_sentiment("This is bad and awful"))
print(simple_sentiment("This product is ok, I will buy again"))