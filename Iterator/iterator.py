from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

import numpy as np


class IteratorBase(ABC):
    """
    Data Iterator class that encapsulates all
    """

    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def replace(self, data, cur_rec):
        self.dataset.replace(data, cur_rec)


class StreamIteratorBase(IteratorBase, ABC):
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
        self.ind = self.get_next()

        train = self.data[from_id: self.ind + 1]
        test = self.data[self.ind + 1]

        self.ind += 1
        return train, test

    @abstractmethod
    def get_next(self):
        pass

    def replace(self, data, rev):
        data = deepcopy(data)
        rev_name = self.dataset.get_revname()
        l = len(data[rev_name])
        data[rev_name][np.random.randint(l)] = rev

        return data


class StreamUntilIterator(StreamIteratorBase):
    def __init__(self, dataset, until_type='pull'):
        super().__init__(dataset)
        self.until_type = until_type

    def get_next(self):
        try:
            while self.data[self.ind + 1]['type'] != self.until_type:
                self.ind += 1
            return self.ind
        except IndexError:
            raise StopIteration


class StreamAllIterator(StreamIteratorBase):
    def get_next(self):
        return self.ind
