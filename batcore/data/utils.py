import re
import numpy as np
from nltk import LancasterStemmer

stemmer = LancasterStemmer()


class ItemMap:
    def __init__(self, data=None):
        if data is None:
            self.id2item = []
            self.item2id = {}
        else:
            self.id2item = list(dict.fromkeys(data))
            self.item2id = {item: i for (i, item) in enumerate(self.id2item)}

    def __getitem__(self, i):
        return self.id2item[i]

    def getid(self, item):
        return self.item2id[item]

    def __len__(self):
        return len(self.id2item)

    def add(self, val):
        self.id2item.append(val)
        self.item2id[val] = len(self.id2item) - 1

    def add2(self, val):
        if val not in self.item2id:
            self.add(val)


def time_interval(col, from_date, to_date):
    """
    :return: column with rows which lies within [from_data; to_date]
    """
    if from_date is None:
        from_date = col.min()
    if to_date is None:
        to_date = col.max()
    return (col >= from_date) & (col <= to_date)


def user_id_split(user_id):
    """
    :return: split user_id into name, email, and login
    """
    id_parts = user_id.split(':')
    name = ':'.join(id_parts[:-2])
    email = id_parts[-2]
    login = id_parts[-1]
    return name, email, login


def get_all_reviewers(events):
    """
    collects all possible reviewers
    """
    reviewer_set = set()
    for event in events:
        if event['type'] == 'pull':
            for reviewer in event["reviewer_login"]:
                reviewer_set.add(reviewer)
    return list(reviewer_set)


def is_word_useful(word):
    """
    word filtering. removes digits and websites
    """
    for c in word:
        if c.isdigit():
            return False
    if "http://" in word or "https://" in word:
        return False
    return True


def word_stem(word):
    """
    removes punctuation and stems with LancasterStemmer
    """
    if word.endswith('.') or word.endswith(',') or word.endswith(':') or word.endswith('\'') or word.endswith('\"'):
        word = word[:-1]
    if word.startswith(',') or word.startswith('.') or word.startswith(':') or word.startswith('\'') or word.startswith(
            '\"'):
        word = word[1:]
    return stemmer.stem(word)


def split_text(txt):
    """
    Splits text, filters and stems words
    """
    splitted_words = list(
        map(lambda x: word_stem(x),
            filter(lambda x: is_word_useful(x), re.split(r"[\s\n\t]+", txt))
            )
    )
    return splitted_words


def get_all_words(events):
    """
    gets list all possible words in reviews
    """
    s = set()
    for event in events:
        if event['type'] == 'pull':
            for w in split_text(event["title"]):
                s.add(w)
    l = list(s)
    return l


def is_bot(x, project=''):
    """
    Filter function for non-human contributors
    """
    name, _, login = user_id_split(x)
    mask_word = re.compile('(bot|test|jenkins|zuul|automation|build|job|infra)', re.IGNORECASE)
    mask_ci = re.compile('ci', re.IGNORECASE)
    mask_pr = re.compile(project, re.IGNORECASE)

    name_ent = name
    if name_ent is np.nan:
        name_ent = login

    if name_ent is np.nan:
        return False

    if len(re.findall(mask_pr, name_ent)):
        return True

    if len(re.findall(mask_word, name_ent)):
        return True

    if len(re.findall(mask_ci, name_ent)):
        return True

    return False
