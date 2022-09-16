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


def time_interval(col, from_date, to_date):
    return (col >= from_date) & (col <= to_date)


def user_id_split(user_id):
    id_parts = user_id.split(':')
    fp = ':'.join(id_parts[:-1])
    sp = id_parts[-1]
    return fp, sp, user_id
