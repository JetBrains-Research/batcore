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
    return common_path


class ItemMap:
    def __init__(self, data):
        self.id2item = list(dict.fromkeys(data))
        self.item2id = {item: i for (i, item) in enumerate(self.id2item)}

    def __getitem__(self, i):
        return self.id2item[i]

    def getid(self, item):
        return self.item2id[item]

    def __len__(self):
        return len(self.id2item)
