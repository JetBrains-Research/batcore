import re

from nltk import LancasterStemmer

stemmer = LancasterStemmer()


def get_all_reviewers(reviews):
    reviewer_set = set()
    for review in reviews:
        for reviewer in review["reviewer_login"]:
            reviewer_set.add(reviewer)
    return list(reviewer_set)


def is_word_useful(word):
    for c in word:
        if c.isdigit():
            return False
    if "http://" in word or "https://" in word:
        return False
    return True


def word_stem(word):
    if word.endswith('.') or word.endswith(',') or word.endswith(':') or word.endswith('\'') or word.endswith('\"'):
        word = word[:-1]
    if word.startswith(',') or word.startswith('.') or word.startswith(':') or word.startswith('\'') or word.startswith(
            '\"'):
        word = word[1:]
    return stemmer.stem(word)


def split_text(txt):
    splitted_words = list(
        map(lambda x: word_stem(x),
            filter(lambda x: is_word_useful(x), re.split(r"[\s\n\t]+", txt))
            )
    )
    return splitted_words


def get_all_words(reviews):
    s = set()
    for review in reviews:
        for w in split_text(review["title"]):
            s.add(w)
    l = list(s)
    return l


def get_map(L):
    return {e: i for i, e in enumerate(L)}
