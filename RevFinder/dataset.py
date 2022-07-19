from collections import defaultdict
from copy import deepcopy
from datetime import timedelta

import numpy as np

from Dataset.dataset import DatasetBase
from Dataset.iterator import RevIterator


class RevFinderDataset(DatasetBase):
    def __init__(self, dataset, initial_delta=None, test_interval=None):
        self.start_date = None
        self.end_date = None
        super(RevFinderDataset, self).__init__(dataset)
        self.log = defaultdict(lambda: [])

    def preprocess(self, dataset):
        pulls = dataset.pulls[dataset.pulls.merged == True]
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'author_login']].rename(
            {'created_at': 'date'}, axis=1)
        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'author_login']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]

        self.id2file = list(dict.fromkeys(pulls.file_path.sum()))
        self.file2id = {f: i for i, f in enumerate(self.id2file)}

        return pulls


class BatchIterator(RevIterator):
    def __init__(self, dataset, initial_delta, test_interval):
        super().__init__(dataset)
        self.initial_delta = timedelta(initial_delta, 0)
        self.test_interval = timedelta(test_interval, 0)

        self.start_date = self.dataset.data.date.min()
        self.end_date = self.dataset.data.date.max()

    def set_params(self, initial_delta, test_interval):
        self.initial_delta = timedelta(initial_delta, 0)
        self.test_interval = timedelta(test_interval, 0)

    def __iter__(self):
        self.to_date = self.initial_delta + self.start_date
        self.test_date = self.to_date + self.test_interval

        self.train = self.dataset.data[
            (self.dataset.data.date < self.to_date) & (self.dataset.data.date >= self.start_date)]
        self.test = self.dataset.data[
            (self.dataset.data.date >= self.to_date) & (self.dataset.data.date < self.test_date)]

        return self

    def __next__(self):
        train = self.train
        test = self.test

        self.train = self.test
        self.test = self.dataset.data[
            (self.dataset.data.date >= self.to_date) & (self.dataset.data.date < self.test_date)]

        self.to_date = self.test_date
        self.test_date = self.to_date + self.test_interval
        if self.to_date > self.end_date:
            raise StopIteration
        return train, test


class OneByOneIterator(RevIterator):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.data = dataset.data.to_dict('records')

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind + 1 >= len(self.data):
            raise StopIteration
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
