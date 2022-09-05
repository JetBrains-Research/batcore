import numpy as np
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

    if full_name_score == 1:
        return 1

    # s1.5

    part_name_score = get_norm_levdist(u1['first_name'], u2['first_name']) + get_norm_levdist(u1['last_name'],
                                                                                              u2['last_name'])
    part_name_score /= 2

    if part_name_score == 1:
        return 1

    # s2

    email_name_score = max(name_email_dist((u1['first_name'], u1['last_name']),
                                           u2['email']),
                           name_email_dist((u2['first_name'], u2['last_name']),
                                           u1['email'])
                           )

    if email_name_score == 1:
        return 1

    # s3

    email_score = 0
    if len(u1['short_email']) > 2 and len(u2['short_email']) > 2:
        email_score = get_norm_levdist(u1['email'], u2['email'])

    return max(full_name_score, part_name_score, email_name_score, email_score)


def get_sim_matrix(users):
    sim_matrix = np.zeros((len(users), len(users)))
    for i1, row1 in users.iterrows():
        for i2, row2 in users.iterrows():
            if i1 > i2:
                score = sim_users(row1, row2)
                sim_matrix[i1, i2] = 1 - score
                sim_matrix[i2, i1] = 1 - score
            if i1 == i2:
                sim_matrix[i1, i2] = 0
    return sim_matrix