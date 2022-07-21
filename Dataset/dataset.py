import os
from abc import ABC, abstractmethod

import pandas as pd


class DatasetBase(ABC):
    """
    Base class for a dataset that encapsulates data preprocessing
    """

    def __init__(self, data):
        """
        :param data: Data to be stored in a dataset
        """
        self.data = self.preprocess(data)

    @abstractmethod
    def preprocess(self, data):
        """
        performs any necessary preprocessing needed
        """
        pass

    def replace(self, data, cur_rec):
        """
        A method that is used for simulating history
        :param data: a single pull that needed to ba modified
        :param cur_rec: reviewer to be added into a pull
        :return: pull with a modified reviewer list
        """
        raise NotImplementedError()


class GithubDataset:
    """
    Helping dataset class for the github-based projects. It reads data in the format that is provided with our
    download script and outputs in a comfortable format
    """

    def __init__(self, path):
        data = GithubDataset.get_df(path)
        self.pulls, self.commits = GithubDataset.prepare(data)

    @staticmethod
    def get_df(path):
        """
        :param path: path to the directory with with csv files
        :return: dictionary with all dataframes
        """
        dfs = {}
        for df in os.listdir(path):
            if df.endswith('.csv'):
                dfs[df.split('.')[0]] = pd.read_csv(path + f'/{df}', sep='|')
        return dfs

    @staticmethod
    def prepare(data):
        """
        :param data: dictionary with all dataframes
        :return: pulls and commits dataframes with all mined features
        """
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
    """
       Helping dataset class for the gerrit-based projects. It reads data in the format that is provided with our
       download script and outputs in a comfortable format
       """

    def __init__(self, path):
        data = GerritDataset.get_df(path)
        self.pulls, self.commits = GerritDataset.prepare(data)

    @staticmethod
    def get_df(path):
        """
        :param path: path to the directory with with csv files
        :return: dictionary with all dataframes for pulls and commits
        """
        data = {}
        for d in ['changes', 'changes_files', 'changes_reviewer', 'commits', 'commits_file', 'commits_author']:
            cur_data = []
            for root, subdirs, files in os.walk(path + f'/{d}'):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.csv'):
                        cur_data.append(pd.read_csv(file_path, sep='|'))
            data[d] = pd.concat(cur_data, axis=0).reset_index()

        return data

    @staticmethod
    def prepare(data):
        """
        :param data: dictionary with all dataframes
        :return: pulls and commits dataframes with all mined features
        """

        # pulls part
        data['changes_reviewer'] = data['changes_reviewer'].drop_duplicates()
        data['changes_files'] = data['changes_files'].drop_duplicates()
        data['changes'] = data['changes'].drop_duplicates()

        pulls = data['changes'].merge(data['changes_files'],
                                      left_on='key_change',
                                      right_on='key_change', how='inner').merge(data['changes_reviewer'],
                                                                                left_on='key_change',
                                                                                right_on='key_change',
                                                                                how='inner')
        pulls = pulls.drop(
            ['index_x', 'index_y', 'index'], axis=1)

        pulls['updated_at'] = pd.to_datetime(pulls.updated_time).dt.tz_localize(None)
        pulls['created_at'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)

        pulls['key_file'] = pulls['key_file'].apply(lambda x: x.replace(':', '/'))
        pulls = pulls.rename(
            {'key_file': 'file_path', 'subject': 'body', 'key_user': 'reviewer_login', 'key': 'number'}, axis=1)

        pulls['comment'] = pulls.comment.apply(lambda x: x.split('Reviewed-on')[0])
        # commits part

        commits = data['commits'].merge(data['commits_file'],
                                        left_on='key',
                                        right_on='key_commit').merge(data['commits_author'],
                                                                     left_on='key',
                                                                     right_on='key_commit')

        commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)
        commits = commits.drop(
            ['oid', 'index_x', 'key', 'index_y', 'key_commit_x', 'index', 'key_commit_y', 'committed_date'], axis=1)

        return pulls, commits
