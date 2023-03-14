from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class LoaderBase(ABC):
    """
    Separate data iterator class that encapsulates iteration logic
    """

    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class StreamLoaderBase(LoaderBase, ABC):
    """
    iterator for time sorted events
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        data = []
        for event_type in dataset.data:
            data += dataset.data[event_type].to_dict('records')
        data = sorted(data, key=lambda x: x['date'])
        self.data = deepcopy(data)

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

        return train, test

    @abstractmethod
    def get_next(self):
        """
        :return: last index of the training point
        """
        pass

    def replace(self, rev):
        l = len(self.data[self.ind + 1]['reviewer'])
        self.data[self.ind + 1]['reviewer'] = deepcopy(self.data[self.ind + 1]['reviewer'])
        self.data[self.ind + 1]['reviewer'][np.random.randint(l)] = rev

        return self.data[self.ind + 1]


class StreamUntilConditionLoader(StreamLoaderBase):
    """
         Stream iterator that iterates until specified amount of events satisfying a condition is met
     """
    def __init__(self, dataset, condition, batch_size=1):
        super().__init__(dataset)
        self.condition = condition
        self.bs = batch_size

    def get_next(self):
        if self.ind + 1 >= len(self.data):
            raise StopIteration
        og = self.ind
        try:
            cnt = 0
            while cnt < self.bs:
                self.ind += 1
                while not self.condition(self.data[self.ind + 1]):
                    self.ind += 1
                cnt += 1
            return self.ind
        except IndexError:
            raise StopIteration

    def replace(self, rev):
        if self.bs != 1:
            raise NotImplementedError
        else:
            return super().replace(rev)


class PullLoader(StreamUntilConditionLoader):
    """
    Stream iterator that iterates over the data and returns batches with batch_size pull requests with at least one
    reviewer (and all other events that it encountered)
    """

    def __init__(self, dataset, batch_size=1):
        super().__init__(dataset, self._condition, batch_size)

    @staticmethod
    def _condition(pull):
        return pull['type'] == 'pull' and (len(pull['reviewer']) > 0)


class PullLoaderAliasTest(StreamUntilConditionLoader):
    """
    Stream iterator that iterates over the data and returns batches with batch_size pull requests with at least one
    that is not a self review reviewer (and all other events that it encountered)
    """

    def __init__(self, dataset, batch_size=1):
        super().__init__(dataset, self._condition, batch_size)

    @staticmethod
    def _condition(pull):
        return pull['type'] == 'pull' and not pull['self_review'] and (len(pull['reviewer']) > 0)
