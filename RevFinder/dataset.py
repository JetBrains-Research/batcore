from datetime import timedelta

from Dataset.dataset import GithubDataset


class RevFinderDataset(GithubDataset):
    def __init__(self, path):
        self.start_date = None
        self.end_date = None
        super(RevFinderDataset, self).__init__(path)


        self.pulls = self.data

    def set_params(self, initial_delta, test_interval):
        self.initial_delta = timedelta(initial_delta, 0)
        self.test_interval = timedelta(test_interval, 0)

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

    def prepare(self, data):
        data = super(RevFinderDataset, self).prepare(data)
        pulls = data[0].groupby('number')[['file_path', 'reviewer_login', 'date']].agg(
            {'file_path': list, 'reviewer_login': lambda x: list(set(x)), 'date': lambda x: list(x)[0]}).reset_index()

        self.start_date = pulls.date.min()
        self.end_date = pulls.date.max()
        return pulls
