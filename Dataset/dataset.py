import os
from abc import ABC, abstractmethod

import pandas as pd


# TODO add comments
class DatasetBase(ABC):
    def __init__(self, dataset):
        self.data = self.preprocess(dataset)

    @abstractmethod
    def preprocess(self, dataset):
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


class GerritDataset:
    def __init__(self, path):
        data = GerritDataset.get_df(path)
        self.pulls = GerritDataset.prepare(data)

    @staticmethod
    def get_df(path):
        pulls = {}
        for d in ['changes', 'changes_files', 'changes_reviewers']:
            data = []
            for root, subdirs, files in os.walk(path + f'/{d}'):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.csv'):
                        data.append(pd.read_csv(file_path, sep='|'))
            pulls[d] = pd.concat(data, axis=0).reset_index()

        return pulls

    @staticmethod
    def prepare(data):
        pulls = data

        pulls['changes_reviewers'] = pulls['changes_reviewers'].drop_duplicates()
        pulls['changes_files'] = pulls['changes_files'].drop_duplicates()
        pulls['changes'] = pulls['changes'].drop_duplicates()

        pulls_df = pulls['changes'].merge(pulls['changes_files'], left_on='key', right_on='key_change',
                                          how='inner').merge(pulls['changes_reviewers'], left_on='key',
                                                             right_on='key_change', how='inner')

        pulls_df = pulls_df.drop(
            ['index_x', 'project', 'change_id', 'number', 'index_y', 'key_change_x', 'key_change_y', 'index', ], axis=1)

        pulls_df['updated_at'] = pd.to_datetime(pulls_df.updated_at).dt.tz_localize(None)
        pulls_df['created_at'] = pd.to_datetime(pulls_df.created_at).dt.tz_localize(None)

        pulls_df['key_file'] = pulls_df['key_file'].apply(lambda x: x.replace(':', '/'))
        pulls_df = pulls_df.rename(
            {'key_file': 'file_path', 'subject': 'body', 'key_user': 'reviewer_login', 'key': 'number'}, axis=1)

        return pulls_df
