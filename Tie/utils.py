import re

import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


def tokenize(text, tokenizer, stemmer):
    if text is np.nan:
        return []
    text = re.sub(r'[^a-zA-Z0-9\-\.\s\_/\\]', ' ', text)
    res = []
    for token in tokenizer.tokenize(text):
        if token[-1] == '.':
            token = token[:-1]
        token = token.lower()
        if len(token) > 1:
            stemmed_token = stemmer.stem(token)
            if token not in stopwords.words('english') and stemmed_token not in stopwords.words('english'):
                res.append(stemmed_token)
    return res


def vectorize(tokens, tok2id):
    res = np.zeros(len(tok2id))
    for t in tokens:
        if t in tok2id:
            res[tok2id[t]] += 1
    return np.array(res)


def revsim(path1, path2):
    parts1 = path1.split('/')
    parts2 = path2.split('/')

    # common = Counter(parts1) & Counter(parts2)
    # common = sum(common.values())
    common = set(parts1).intersection(set(parts2))  # maybe counter instead of set

    return len(common) / max(len(parts1), len(parts2))
