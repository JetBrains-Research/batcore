import re
import numpy as np
import pandas as pd
from nltk import LancasterStemmer


from aliasmatching import BirdMatching
# from batcore.alias.utils import get_clusters

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
            for reviewer in event["reviewer"]:
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


def preprocess_users(data, remove_bots, bots, factorize_users, alias, project_name, threshold=0.1):
    u1 = pd.unique(data.pulls.owner.sum())
    u2 = pd.unique(data.pulls.reviewer.sum())
    u3 = pd.unique(data.commits.key_user)
    u4 = pd.unique(data.comments.key_user)
    u5 = pd.unique(data.pulls.author.apply(lambda x: list(x)).sum())

    users = np.unique(np.hstack((u1, u2, u3, u4, u5)))

    if remove_bots:
        if bots != 'auto':
            bots = pd.read_csv(bots).fillna('')
            bots = bots.apply(lambda x: f'{x["name"]}:{x["email"]}:{x["login"]}', axis=1)
            bots = set(bots)
        else:
            bots = set([u for u in users if is_bot(u, project_name)])

        users = np.array([u for u in users if u not in bots])

        data.pulls['owner'] = data.pulls['owner'].apply(lambda x: [u for u in x if u not in bots])
        data.pulls['reviewer'] = data.pulls['reviewer'].apply(lambda x: [u for u in x if u not in bots])
        data.pulls['author'] = data.pulls['author'].apply(lambda x: [u for u in x if u not in bots])

        data.pulls = data.pulls[data.pulls.owner.apply(lambda x: len(x) > 0)]
        data.pulls = data.pulls[data.pulls.reviewer.apply(lambda x: len(x) > 0)]

        data.commits['key_user'] = data.commits['key_user'].apply(lambda x: np.nan if x in bots else x)
        data.comments['key_user'] = data.comments['key_user'].apply(lambda x: np.nan if x in bots else x)

        data.comments = data.comments[~data.comments.key_user.isna()]
        data.commits = data.commits[~data.commits.key_user.isna()]

    if factorize_users:
        if alias:
            users_parts = [user_id_split(s) for s in users]

            users_df = pd.DataFrame({'email': [u[1] for u in users_parts],
                                     'name': [u[0] for u in users_parts],
                                     'login': [u[2] for u in users_parts],
                                     'initial_id': [u for u in users]})

            clusters = BirdMatching(distance_threshold=threshold).get_clusters(users_df)
        else:
            clusters = {u: i for i, u in enumerate(users)}

        data.clusters = clusters
        data.pulls.owner = data.pulls.owner.apply(lambda x: list(set([clusters[u] for u in x])))
        data.pulls.author = data.pulls.author.apply(lambda x: list(set([clusters[u] for u in x])))
        data.pulls.reviewer = data.pulls.reviewer.apply(lambda x: list(set([clusters[u] for u in x])))

        data.commits.key_user = data.commits.key_user.apply(lambda x: clusters[x])
        data.comments.key_user = data.comments.key_user.apply(lambda x: clusters[x])

        return clusters


def add_self_review(data_a, data_f):
    try:
        data_f.pulls.drop('self_review', axis=1)
    except KeyError:
        pass

    data_f.pulls['self_review'] = data_a.pulls.apply(lambda x:
                                                     len(set(x.reviewer).intersection(set(x.author))) > 0,
                                                     axis=1)
