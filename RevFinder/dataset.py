from collections import defaultdict

from Dataset.dataset import DatasetBase


class RevFinderDataset(DatasetBase):
    def __init__(self, dataset, initial_delta=None, test_interval=None):
        self.start_date = None
        self.end_date = None
        super(RevFinderDataset, self).__init__(dataset, initial_delta, test_interval)
        self.log = defaultdict(lambda: [])
        self.pulls = self.data

    def preprocess(self, dataset):
        pulls = dataset.pulls[dataset.pulls.merged == True]
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'author_login']].rename(
            {'created_at': 'date'}, axis=1)
        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'author_login']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        self.start_date = pulls.date.min()
        self.end_date = pulls.date.max()

        self.id2file = list(set(pulls.file_path.sum()))
        self.file2id = {f: i for i, f in enumerate(self.id2file)}

        return pulls

    def __iter__(self):
        self.to_date = self.initial_delta + self.start_date
        self.test_date = self.to_date + self.test_interval

        self.train = self.pulls[(self.pulls.date < self.to_date) & (self.pulls.date >= self.start_date)]
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        return self

    def __next__(self):
        train = self.train
        test = self.test

        self.train = self.test
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        self.to_date = self.test_date
        self.test_date = self.to_date + self.test_interval
        if self.to_date > self.end_date:
            raise StopIteration
        return train, test
