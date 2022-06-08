import os
from abc import ABC, abstractmethod

import pandas as pd


# TODO add comments
class DatasetBase(ABC):
    @abstractmethod
    def set_params(self, initial_delta, test_interval):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class GithubDataset(DatasetBase):
    def __init__(self, path):
        dfs = GithubDataset.get_df(path)
        self.pulls, self.commits = self.prepare(dfs)

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

    def prepare(self, data):
        pulls = data['pull_file'].merge(data['reviewer']).merge(data['pull'][['number', 'created_at']],
                                                                left_on='pull_number',
                                                                right_on='number').merge(data['pull_author']).dropna()
        pulls['date'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)
        commits = data['commit_file'].merge(data['commit_author']).merge(data['commit'][['oid', 'committed_date']],
                                                                         left_on='oid', right_on='oid').dropna()
        commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)

        pulls = pulls.drop(['pull_number', 'created_at'], axis=1)
        commits = commits.drop(['oid', 'file_oid', 'committed_date'], axis=1)

        return pulls, commits
