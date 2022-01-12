import numpy as np


class ObjectContainer:
    def __init__(self, objects):
        self.id2obj = list(objects)
        self.obj2id = {o: i for i, o in enumerate(objects)}

        self.contain_mask = np.zeros(len(objects), dtype=int)

    def __len__(self):
        return len(self.id2obj)

    def contain(self, obj):
        return self.contain_mask[obj] == 1

    def get_ids(self, objs):
        return np.array([self.obj2id[obj] for obj in objs], dtype=int)

    def get_id(self, obj):
        return self.obj2id[obj]
