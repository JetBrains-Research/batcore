from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

import numpy as np


class IteratorBase(ABC):
    """
    Separate data Iterator class that encapsulates
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
    """
    Iterator for time sorted events
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.data = dataset.data

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        """
        Iterates over the data stream until index from get_next method
        :return: a pair of train and test data
        """
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
        """
        :return: last index of the training point
        """
        pass

    def replace(self, data, rev):
        data = deepcopy(data)
        rev_name = self.dataset.get_revname()
        l = len(data[rev_name])
        data[rev_name][np.random.randint(l)] = rev

        return data


class StreamUntilIterator(StreamIteratorBase):
    """
    Stream iterator that iterates until next event of the certain type
    """

    def __init__(self, dataset, until_type='pull'):
        super().__init__(dataset)
        self.until_type = until_type

    def get_next(self):
        """
        :return: index before the event with until_type
        """
        try:
            while self.data[self.ind + 1]['type'] != self.until_type:
                self.ind += 1
            return self.ind
        except IndexError:
            raise StopIteration


class StreamAllIterator(StreamIteratorBase):
    """
    Stream iterator that iterates over each event
    """

    def get_next(self):
        return self.ind


class BatchStreamIterator(StreamIteratorBase):
    """
    Stream iterator that iterates until specified amount of events of the certain type are encountered
    """

    def __init__(self, dataset, batch_size=10, until_type='pull'):
        super().__init__(dataset)
        self.until_type = until_type
        self.bs = batch_size

    def get_next(self):
        if self.ind + 1 >= len(self.data):
            raise StopIteration
        og = self.ind
        try:
            cnt = 0
            while cnt < self.bs:
                self.ind += 1
                while self.data[self.ind + 1]['type'] != self.until_type:
                    self.ind += 1
                cnt += 1
            return self.ind
        except IndexError:
            raise StopIteration
