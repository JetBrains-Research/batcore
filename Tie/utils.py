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


def revsim(path1, path2):
    parts1 = path1.split('/')
    parts2 = path2.split('/')

    common = set(parts1).intersection(set(parts2))  # maybe counter instead of set

    return len(common) / max(len(parts1), len(parts2))
