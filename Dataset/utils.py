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
