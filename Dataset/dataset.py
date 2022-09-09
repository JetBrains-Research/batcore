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


class GerritDataset:
    """
       Helping dataset class for the gerrit-based projects. It reads data in the format that is provided with our
       download script and outputs in a comfortable format
       """

    def __init__(self, path, from_checkpoint=False):
        if from_checkpoint:
            self.pulls = pd.read_csv(path + '/pulls.csv')
            self.pulls.created_at = pd.to_datetime(self.pulls.created_at).dt.tz_localize(None)
            self.commits = pd.read_csv(path + '/commits.csv')
            self.commits.date = pd.to_datetime(self.commits.date).dt.tz_localize(None)
            self.comments = pd.read_csv(path + '/comments.csv')
            self.comments.date = pd.to_datetime(self.comments.date).dt.tz_localize(None)

            return
        data = GerritDataset.get_df(path)
        self.pulls, self.commits, self.comments = GerritDataset.prepare(data)

    @staticmethod
    def get_df(path):
        """
        :param path: path to the directory with csv files
        :return: dictionary with all dataframes for pulls and commits
        """
        data = {}
        for d in ['changes', 'changes_files', 'changes_reviewer', 'commits', 'commits_file', 'commits_author',
                  'comments_file', 'comments_patch']:
            cur_data = []
            for root, subdirs, files in os.walk(path + f'/{d}'):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.csv'):
                        cur_data.append(
                            pd.read_csv(file_path, sep='|', on_bad_lines='skip'))  # todo remove on bad lines
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
            {'key_file': 'file_path', 'subject': 'body', 'key_user_y': 'reviewer_login', 'key': 'number',
             'key_user_x': 'owner'}, axis=1)

        pulls.comment = pulls.comment.fillna('')
        pulls['comment'] = pulls.comment.apply(lambda x: x.split('Reviewed-on')[0])

        # pulls = pulls[pulls.reviewer_login != 'Jenkins:']
        # commits part

        commits = data['commits'].merge(data['commits_file'],
                                        left_on='key_commit',
                                        right_on='key_commit').merge(data['commits_author'],
                                                                     left_on='key_commit',
                                                                     right_on='key_commit')

        commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)

        commits = commits.drop(
            ['oid', 'index_x', 'index_y', 'index', 'committed_date', 'lines_inserted', 'lines_deleted', 'size',
             'size_delta'], axis=1)
        commits['key_file'] = commits['key_file'].apply(lambda x: x.replace(':', '/'))

        # comments part
        comments_file = data['comments_file']
        comments_file.time = pd.to_datetime(comments_file.time).dt.tz_localize(None)
        comments_file = comments_file.drop(['index'], axis=1).rename({'time': 'date'}, axis=1)
        comments_file['key_file'] = comments_file['key_file'].apply(lambda x: x.replace(':', '/'))

        comments_pull = data['comments_patch']
        comments_pull.time = pd.to_datetime(comments_pull.time).dt.tz_localize(None)
        comments_pull = comments_pull.drop(['index', 'oid'], axis=1).rename({'time': 'date'}, axis=1)
        comments_pull['key_file'] = None

        comments = pd.concat((comments_pull, comments_file), axis=0).reset_index().drop(['index'], axis=1)

        return pulls, commits, comments


