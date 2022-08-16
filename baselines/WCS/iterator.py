from copy import deepcopy

import numpy as np

from Iterator.iterator import IteratorBase


class StreamIterator(IteratorBase):
    """
    iterates over time stream to the next review
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.data = dataset.data

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind + 1 >= len(self.data):
            raise StopIteration

        from_id = self.ind

        train = self.data[self.ind]
        test = self.data[self.ind + 1]

        self.ind += 1
        return train, test

    def replace(self, data, rev):
        data = deepcopy(data)
        rev_name = self.dataset.get_revname()
        l = len(data[rev_name])
        data[rev_name][np.random.randint(l)] = rev

        return data
