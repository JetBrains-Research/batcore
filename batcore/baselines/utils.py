def get_map(L):
    return {e: i for i, e in enumerate(L)}


def pull_sim(pull1, pull2):
    """
    counts file path-based similarity for pull1 and pull2
    """
    changed_files1 = pull1["file"]
    changed_files2 = pull2["file"]
    if len(changed_files1) == 0 or len(changed_files2) == 0:
        return 0
    sum_score = 0
    for f1 in changed_files1:
        s1 = set(f1.split('/'))
        for f2 in changed_files2:
            s2 = set(f2.split('/'))
            sum_score += (len(s1 & s2)) / max(len(s1), len(s2))
    ret = sum_score / (len(changed_files1) * len(changed_files2) + 1)
    return ret


def camel_split(path):
    """
    :param path: file path
    :return: tokens from path split by '/' and camel case
    """
    tokens = []
    cur_token = ""
    for c in path:
        if c == '/':
            tokens.append(cur_token)
            cur_token = ""
        elif c.isupper():
            tokens.append(cur_token)
            cur_token = c
        else:
            cur_token += c
    return tokens


def sim(f1, f2):
    """
    :param f1: file path1
    :param f2: file path2
    :return: similarity measure between files
    """
    t1 = set(camel_split(f1))
    t2 = set(camel_split(f2))
    return len(t1.intersection(t2)) / len(t1.union(t2))


def norm(p):
    """
    :param p: list of scores
    :return: min-max score normalizarion
    """
    p -= p.min()
    if p.max() == 0:
        return p
    return p / p.max()


# from https://github.com/patanamon/revfinder

#########################################################################
# File: stringCompare.py
# Descriptions: String comparison techniques for file path similarity
# Input: The arguments f1, f2 are strings of file path
# Output: Number of common file path components in f1 and f2
# Written By: Patanamon Thongtanunam (patanamon-t@is.naist.jp)
#########################################################################

def path2list(fileString):
    return fileString.split("/")


def LCP(f1, f2):
    """
    Longest Common Prefix
    """
    common_path = 0
    min_length = min(len(f1), len(f2))
    for i in range(min_length):
        if f1[i] == f2[i]:
            common_path += 1
        else:
            break
    return common_path


def LCSuff(f1, f2):
    """Longest Common Suffiz"""
    common_path = 0
    r = range(min(len(f1), len(f2)))
    for i in reversed(r):
        if f1[i] == f2[i]:
            common_path += 1
        else:
            break
    return common_path


def LCSubstr(f1, f2):
    """Longest Common Substring"""
    common_path = 0
    if len(set(f1) & set(f2)) > 0:
        mat = [[0 for _ in range(len(f2) + 1)] for _ in range(len(f1) + 1)]
        for i in range(len(f1) + 1):
            for j in range(len(f2) + 1):
                if i == 0 or j == 0:
                    mat[i][j] = 0
                elif f1[i - 1] == f2[j - 1]:
                    mat[i][j] = mat[i - 1][j - 1] + 1
                    common_path = max(common_path, mat[i][j])
                else:
                    mat[i][j] = 0
    return common_path


def LCSubseq(f1, f2):
    """Longest Common Subsequence"""
    if len(set(f1) & set(f2)) > 0:
        L = [[0 for x in range(len(f2) + 1)] for x in range(len(f1) + 1)]
        for i in range(len(f1) + 1):
            for j in range(len(f2) + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif f1[i - 1] == f2[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        common_path = L[len(f1)][len(f2)]
    else:
        common_path = 0
    return common_path
