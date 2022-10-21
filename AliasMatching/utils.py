import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from Levenshtein import distance as LevDist
import string
from tqdm import tqdm
from collections import defaultdict


def remove_punctuation(name):
    """
    removes
    """
    return name.translate(str.maketrans('', '', string.punctuation)).lower()


# prefixes and titles to remove from name
fixes = ['jr', 'sr', 'dr', 'mr', 'mrs']
titles = ['admin', 'support']
ban_words = fixes + titles


def remove_ban_words(name):
    """
    removes banned words from name
    """
    name_parts = []
    for p in name.split():
        if p not in ban_words:
            name_parts.append(p)
    return " ".join(name_parts)


def name_preprocess(name):
    """
    removes punctuation and banned words from the name
    """
    return remove_ban_words(remove_punctuation(name))


def first_name(name):
    """
    returns first name
    """
    if name == '':
        return name
    return name.split()[0]


def last_name(name):
    """
    return last name
    """
    if name == '':
        return ''
    return name.split()[-1]


def shorten_email(email):
    """
    returns shorten form of the e-mail (everything before @)
    """
    if email == '':
        return ''
    return email.split('@')[0]


def get_norm_levdist(str1, str2):
    """
    Normalised Levenshtein distance between two strings
    """
    ld = LevDist(str1, str2)
    ml = max(len(str1), len(str2))
    score = ld / ml

    return score


def name_email_dist(name, email):
    """
    checks if first and second name are in e-mail
    """
    fn, ln = name
    if len(fn) > 1 and len(ln) > 1:
        if fn in email and ln in email:
            return 0

    # if len(fn) > 0 and len(ln) > 0:
    #     names_with_initials = [fn[0] + ln, fn + ln[0], ln + fn[0], ln[0] + fn]
    #     for nwi in names_with_initials:
    #         if len(nwi) > 2 and nwi in email:
    #             return 0
    return 1


def sim_users(u1, u2):
    """
    similarity of two users name based on their name and e-mail
    """
    # s1
    #     return (u1, u2)
    full_name_score = get_norm_levdist(u1['name'], u2['name'])

    #     if full_name_score == 1:
    #         return 1

    # s1.5

    part_name_score = get_norm_levdist(u1['first_name'], u2['first_name']) + get_norm_levdist(u1['last_name'],
                                                                                              u2['last_name'])
    part_name_score /= 2

    #     if part_name_score == 1:
    #         return 1

    # s2

    email_name_score = max(name_email_dist((u1['first_name'], u1['last_name']),
                                           u2['email']),
                           name_email_dist((u2['first_name'], u2['last_name']),
                                           u1['email'])
                           )

    #     if email_name_score == 1:
    #         return 1

    # s3

    email_score = 1
    if not u1['short_email'] is np.nan and not u2['short_email'] is np.nan:
        if len(u1['short_email']) > 2 and len(u2['short_email']) > 2:
            email_score = get_norm_levdist(u1['email'], u2['email'])

    return min(full_name_score, part_name_score, email_name_score, email_score)


def ban_users(x):
    """
    Filter function for non-human contributors
    """
    if 'ci' in x['name'].split():
        return False

    if 'bot' in x['name'].split():
        return False

    if x['name'] in ['jenkins', 'zuul', 'welcome new contributor']:
        return False

    return True


# def get_score_func(i1, row1):
#     def score_func(i2, row2):
#         nonlocal i1, row1
#         if i1 > i1:
#             return sim_users(row1, row2)
#         if i1 == i2:
#             return 0
#
#     return score_func


def get_sim_matrix(users):
    """
    calculates name similarity matrix between users
    """
    sim_matrix = np.zeros((len(users), len(users)))
    for i1, row1 in tqdm(users.iterrows()):
        def score(i2, row2):
            if i1 < i2:
                return sim_users(row1, row2)
            if i1 == i2:
                return 0
            return 0

        # sim_matrix[i1] = pool.starmap(score, users.iterrows())
        sim_matrix[i1] = [score(*p) for p in users.iterrows()]
    sim_matrix = sim_matrix + sim_matrix.T
    # for i2, row2 in users.iterrows():
    #     if i1 > i2:
    #         score = sim_users(row1, row2)
    #         sim_matrix[i1, i2] = score
    #         sim_matrix[i2, i1] = score
    #     if i1 == i2:
    #         sim_matrix[i1, i2] = 0
    return sim_matrix


def get_clusters(users, distance_threshold=0.1):
    """
    Adaptation of the approach from the 'Mining Email Social Networks'. Algorithm measures pair-wise similarity
    between all participants and splits them into clusters with Agglomerative clustering.
    :param users: dataframe with users names, e-mails, and logins
    :param distance_threshold: distance parameter for clustering
    :return: dict with provides cluster id for each user
    """
    users = users.drop_duplicates().reset_index().drop('index', axis=1)
    users = users.fillna('')
    users.name = users.name.apply(name_preprocess)

    users['short_email'] = users.email.apply(shorten_email)
    users['first_name'] = users.name.apply(first_name)
    users['last_name'] = users.name.apply(last_name)

    users = users[users.apply(ban_users, axis=1)]
    users = users.reset_index().drop('index', axis=1)

    sim_matrix = get_sim_matrix(users)
    agg = AgglomerativeClustering(n_clusters=None,
                                  distance_threshold=distance_threshold,
                                  affinity='precomputed',
                                  linkage='complete').fit(sim_matrix)
    cl, cn = np.unique(agg.labels_, return_counts=True)

    # sind = np.argsort(-cn)
    # cl = cl[sind]
    # cn = cn[sind]

    users['cluster'] = agg.labels_
    df_cs = users[['cluster', 'name']].groupby('cluster').count().reset_index().rename({'name': 'cluster_size'},
                                                                                       axis=1)

    users = users.join(df_cs, on='cluster', rsuffix='_r').drop('cluster_r', axis=1)
    users = users.sort_values(['cluster_size', 'cluster'], ascending=False).reset_index().drop('index', axis=1)

    users['cluster2'] = -users.index - 1

    users.loc[users['cluster'].isna(), 'cluster'] = users['cluster2'][users['cluster'].isna()]
    users['id'] = pd.factorize(users['cluster'])[0]

    key2id = {str(x['initial_id']): x['id'] for _, x in users.iterrows()}
    key2id = defaultdict(lambda: np.nan, key2id)
    return key2id
