import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from Levenshtein import distance as LevDist
import string


def remove_punctuation(name):
    return name.translate(str.maketrans('', '', string.punctuation)).lower()


fixes = ['jr', 'sr', 'dr', 'mr', 'mrs']
titles = ['admin', 'support']
ban_words = fixes + titles


def remove_banwords(name):
    name_parts = []
    for np in name.split():
        if np not in ban_words:
            name_parts.append(np)
    return " ".join(name_parts)


def name_preprocess(name):
    return remove_banwords(remove_punctuation(name))


def first_name(name):
    return name.split()[0]


def last_name(name):
    return name.split()[-1]


def shorten_email(email):
    return email.split('@')[0]


def get_norm_levdist(str1, str2):
    #     str1 = str(str1)
    #     str2 = str(str2)

    ld = LevDist(str1, str2)
    ml = max(len(str1), len(str2))
    score = (ml - ld) / ml

    return score


def name_email_dist(name, email):
    fn, ln = name
    if len(fn) > 1 and len(ln) > 1:
        if fn in email and ln in email:
            return 1

    names_with_initials = [fn[0] + ln, fn + ln[0], ln + fn[0], ln[0] + fn]
    for nwi in names_with_initials:
        if len(nwi) > 2 and nwi in email:
            return 1
    return 0


def sim_users(u1, u2):
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
    if 'ci' in x['name'].split():
        return False

    if 'bot' in x['name'].split():
        return False

    if x['name'] in ['jenkins', 'zuul', 'welcome new contributor']:
        return False

    return True


def get_sim_matrix(users):
    sim_matrix = np.zeros((len(users), len(users)))
    for i1, row1 in users.iterrows():
        for i2, row2 in users.iterrows():
            if i1 > i2:
                score = sim_users(row1, row2)
                sim_matrix[i1, i2] = score
                sim_matrix[i2, i1] = score
            if i1 == i2:
                sim_matrix[i1, i2] = 0
    return sim_matrix


def get_clusters(users, name='df_with_clusters'):
    sim_matrix = get_sim_matrix(users)
    agg = AgglomerativeClustering(n_clusters=None,
                                  distance_threshold=0.1,
                                  affinity='precomputed',
                                  linkage='complete').fit(sim_matrix)
    cl, cn = np.unique(agg.labels_, return_counts=True)

    sind = np.argsort(-cn)
    cl = cl[sind]
    cn = cn[sind]

    users['cluster'] = agg.labels_
    df_cs = users[['cluster', 'name']].groupby('cluster').count().reset_index().rename({'name': 'cluster_size'},
                                                                                         axis=1)

    users = users.join(df_cs, on='cluster', rsuffix='_r').drop('cluster_r', axis=1)
    users = users.sort_values(['cluster_size', 'cluster'], ascending=False).reset_index().drop('index', axis=1)

    users.to_csv(f"{name}.csv")
    return users


def get_key2id(path):
    users = pd.read_csv(path)
    users['cluster2'] = -users.index - 1

    users.loc[users['cluster'].isna(), 'cluster'] = users['cluster2'][users['cluster'].isna()]
    users['id'] = pd.factorize(users['cluster'])[0]

    key2id = {str(x['name']) + ':' + str(x['email']): x['id'] for _, x in users.iterrows()}

    return key2id
