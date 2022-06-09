import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


def stem(text, pt):
    if text is np.nan:
        return []
    res = []
    for w in word_tokenize(text):
        if w not in stopwords.words('english'):
            if w.isalpha() or len(w) > 1:
                res.append(pt.stem(w))
    return res
