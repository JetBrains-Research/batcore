import numpy as np


def std(a):
    return np.sqrt((a.multiply(a).sum() / a.shape[0] - (a.sum() / a.shape[0]) ** 2))


def pearson(a, b):
    num = (a.multiply(b)).sum() / a.shape[0] - a.sum() * b.sum() / a.shape[0] / b.shape[0]
    return num / std(a) / std(b)


def LCP(f1, f2):
    f1 = f1.split('/')
    f2 = f2.split('/')

    common_path = 0
    min_length = min(len(f1), len(f2))
    for i in range(min_length):
        if f1[i] == f2[i]:
            common_path += 1
        else:
            break
    return common_path / max(len(f1), len(f2))
