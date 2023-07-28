from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

SW = stopwords.words("english")


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=1, min_length=2):
    """
    Source: https://github.com/iridia-ulb/AI-book/tree/main/SpamDetector
    """

    if lower_case:
        message = str(message).lower()

    words = word_tokenize(message)
    words = [w for w in words if len(w) > min_length]

    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [" ".join(words[i : i + gram])]
        return w

    if stop_words:
        words = [word for word in words if word not in SW]

    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    return words
