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


def camel_split(path):
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
    t1 = set(camel_split(f1))
    t2 = set(camel_split(f2))
    return len(t1.intersection(t2)) / len(t1.union(t2))


def norm(p):
    p -= p.min()
    if p.max() == 0:
        return p
    return p / p.max()
