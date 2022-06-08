import numpy as np

from Dataset.dataset import DatasetBase
from recsys.mapping import MappingWithFallback, Mapping


class RecSysDataset(DatasetBase):
    def __init__(self, dataset, initial_delta=None, test_interval=None):

        self.commit_file_mapping = None
        self.pull_file_mapping = None

        self.commit_user_mapping = None
        self.pull_user_mapping = None

        self.start_date = None
        self.end_date = None

        super(RecSysDataset, self).__init__(dataset, initial_delta, test_interval)

        self.pulls, self.commits = self.data


    def preprocess(self, dataset):
        pulls = dataset.pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'author_login']].rename(
            {'created_at': 'date'}, axis=1)
        commits = dataset.commits[['file_path', 'author_login', 'committed_date']].rename({'committed_date': 'date'},
                                                                                          axis=1)

        id2file = np.unique(np.hstack((pulls.file_path, commits.file_path)))
        id2user = np.unique(np.hstack((pulls.reviewer_login, pulls.author_login, commits.author_login)))

        file2id = {f: i for i, f in enumerate(id2file)}
        user2id = {u: i for i, u in enumerate(id2user)}

        pulls['file_path'] = pulls.file_path.apply(lambda x: file2id[x])
        pulls['reviewer_login'] = pulls.reviewer_login.apply(lambda x: user2id[x])
        pulls['author_login'] = pulls.author_login.apply(lambda x: user2id[x])

        commits['file_path'] = commits.file_path.apply(lambda x: file2id[x])
        commits['author_login'] = commits.author_login.apply(lambda x: user2id[x])

        # TODO make one class with 2 masks
        self.commit_file_mapping = MappingWithFallback(file2id, id2file)
        self.pull_file_mapping = MappingWithFallback(file2id, id2file)

        self.commit_user_mapping = Mapping(file2id, id2file)
        self.pull_user_mapping = Mapping(file2id, id2file)

        for d in [pulls, commits]:
            self.start_date = d.date.min() if self.start_date is None else min(self.start_date, d.date.min())
            self.end_date = d.date.max() if self.end_date is None else max(self.end_date, d.date.max())

        return pulls, commits

    def __iter__(self):
        self.to_date = self.initial_delta + self.start_date
        self.test_date = self.to_date + self.test_interval

        self.train = [df[(df.date < self.to_date) & (df.date >= self.start_date)] for df in [self.pulls, self.commits]]
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        return self

    def __next__(self):

        test = self.test.groupby('number')[['file_path', 'reviewer_login']].agg(
            {'file_path': list, 'reviewer_login': lambda x: list(set(x))}).reset_index()

        pulls = self.train[0].groupby(['file_path', 'reviewer_login', 'author_login']).count().reset_index().drop(
            ['date', 'author_login'], axis=1).rename({'reviewer_login': 'login'}, axis=1)
        commits = self.train[1].groupby(['file_path', 'author_login']).count().reset_index().rename(
            {'date': 'number', 'author_login': 'login'}, axis=1)

        train = (pulls, commits)

        if self.to_date > self.end_date:
            raise StopIteration

        self.train = (
            self.test, self.commits[(self.commits.date >= self.to_date) & (self.commits.date < self.test_date)])
        self.train = (
            self.test, self.commits[(self.commits.date >= self.to_date) & (self.commits.date < self.test_date)])

        self.to_date = self.test_date
        self.test_date = self.to_date + self.test_interval
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        return train, test
