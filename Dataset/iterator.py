from abc import ABC, abstractmethod


class RevIterator(ABC):
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
