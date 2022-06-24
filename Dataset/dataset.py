import os
from abc import ABC, abstractmethod
from datetime import timedelta

import pandas as pd


# TODO add comments
class DatasetBase(ABC):
    def __init__(self, dataset, initial_delta=None, test_interval=None):
        self.data = self.preprocess(dataset)
        self.initial_delta = timedelta(initial_delta, 0)
        self.test_interval = timedelta(test_interval, 0)

    def set_params(self, initial_delta, test_interval):
        self.initial_delta = timedelta(initial_delta, 0)
        self.test_interval = timedelta(test_interval, 0)

    @abstractmethod
    def preprocess(self, dataset):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class GithubDataset:
    def __init__(self, path):
        data = GithubDataset.get_df(path)
        self.pulls, self.commits = GithubDataset.prepare(data)

    @staticmethod
    def get_df(path):
        dfs = {}
        for df in os.listdir(path):
            try:
                dfs[df.split('.')[0]] = pd.read_csv(path + f'/{df}', sep='|')
            except Exception as e:
                # TODO add correct exception
                continue
        return dfs

    @staticmethod
    def prepare(data):
        pulls = data['pull_file'].merge(data['reviewer']).merge(data['pull'],
                                                                left_on='pull_number',
                                                                right_on='number').merge(data['pull_author'])
        pulls['created_at'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)
        pulls['last_edited_at'] = pd.to_datetime(pulls.last_edited_at).dt.tz_localize(None)
        pulls['published_at'] = pd.to_datetime(pulls.published_at).dt.tz_localize(None)
        pulls['updated_at'] = pd.to_datetime(pulls.updated_at).dt.tz_localize(None)
        pulls['merged_at'] = pd.to_datetime(pulls.merged_at).dt.tz_localize(None)
        pulls['closed_at'] = pd.to_datetime(pulls.closed_at).dt.tz_localize(None)
        pulls = pulls.drop(['pull_number'], axis=1)

        commits = data['commit_file'].merge(data['commit_author']).merge(data['commit'],
                                                                         left_on='oid', right_on='oid').dropna()

        commits['committed_date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)
        commits = commits.drop(['oid', 'file_oid'], axis=1)

        return pulls, commits
