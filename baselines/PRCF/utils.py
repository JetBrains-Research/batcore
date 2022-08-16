import numpy as np


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
