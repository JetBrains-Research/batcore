import ast
import os

import numpy as np
import pandas as pd

from batcore.alias.utils import get_clusters
from batcore.data.utils import time_interval, user_id_split, is_bot


class GerritLoader:
    """
        Helping dataset class for the gerrit-based projects. It reads data in the format that is provided with our
        download script and outputs in a comfortable format

        :param path: path to the folder with the data from the loader tool or to the saved dataset
        :param from_date: all events before from_date are removed from the data
        :param to_date: all events after to_date are removed from the data
        :param from_checkpoint: set to True to load saved dataset
       """

    def __init__(self, path,
                 from_date=None,
                 to_date=None,
                 from_checkpoint=False,
                 ):

        if from_checkpoint:
            self.from_checkpoint(path)
        else:

            self.from_date = from_date
            self.to_date = to_date

            # self.factorize = factorize_users

            data = GerritLoader.get_df(path)
            self.pulls, self.commits, self.comments = self.prepare(data)
            self.prepare_pulls()

            # if process_users:
            #     self.prepare_users(remove_bots, bots, factorize_users, alias, project_name)

    @staticmethod
    def get_df(path):
        """
        :param path: path to the directory with csv files
        :return: dictionary with all dataframes for pulls, commits, and comments
        """
        data = {}
        for d in ['changes', 'changes_files', 'changes_reviewer', 'commits', 'commits_file', 'commits_author',
                  'comments_file', 'comments_patch', 'users']:
            cur_data = []
            for root, subdirs, files in os.walk(path + f'/{d}'):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.csv'):
                        cur_data.append(
                            pd.read_csv(file_path, sep='|', on_bad_lines='skip'))  # todo remove on bad lines
            data[d] = pd.concat(cur_data, axis=0).reset_index()

        return data

    def prepare(self, data):
        """
        :param data: dictionary with all dataframes
        :return: pulls and commits dataframes with all mined features
        """
        # commits part

        # unify timezone
        data['commits']['committed_date'] = pd.to_datetime(data['commits'].committed_date).dt.tz_localize(None)

        # remove commits not in the timeframe
        data['commits'] = data['commits'][
            time_interval(data['commits']['committed_date'], self.from_date, self.to_date)]

        # add authors
        commits = data['commits'].merge(data['commits_file'],
                                        left_on='key_commit',
                                        right_on='key_commit').merge(data['commits_author'],
                                                                     left_on='key_commit',
                                                                     right_on='key_commit')

        commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)

        # remove some columns
        commits = commits.drop(
            ['oid', 'index_x', 'index_y', 'index', 'committed_date', 'lines_inserted', 'lines_deleted', 'size',
             'size_delta', 'uploader_key_user', 'committer_key_user'], axis=1)
        # re-format files paths
        commits['key_file'] = commits['key_file'].apply(lambda x: x.replace(':', '/'))

        commits = commits.rename({'author_key_user': 'key_user'}, axis=1)

        # pulls part

        # remove duplicates and unify time
        data['changes_reviewer'] = data['changes_reviewer'].drop_duplicates()
        data['changes_files'] = data['changes_files'].drop_duplicates()
        data['changes'] = data['changes'].drop_duplicates()
        data['changes']['created_at'] = pd.to_datetime(data['changes'].created_at).dt.tz_localize(None)

        # remove pulls not in the time frame
        data['changes'] = data['changes'][time_interval(data['changes']['created_at'], self.from_date, self.to_date)]

        # add more fields
        pulls = data['changes'].merge(data['changes_files'],
                                      left_on='key_change',
                                      right_on='key_change', how='inner').merge(data['changes_reviewer'],
                                                                                left_on='key_change',
                                                                                right_on='key_change',
                                                                                how='inner')

        # data cleaning
        pulls = pulls.drop(
            ['index_x', 'index_y', 'index'], axis=1)

        pulls['updated_at'] = pd.to_datetime(pulls.updated_time).dt.tz_localize(None)
        pulls['created_at'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)

        # reformatting and renaming
        pulls['key_file'] = pulls['key_file'].apply(lambda x: x.replace(':', '/'))
        pulls = pulls.rename(
            {'key_file': 'file_path', 'subject': 'body', 'key_user_y': 'reviewer_login', 'key': 'number',
             'key_user_x': 'owner'}, axis=1)

        pulls.comment = pulls.comment.fillna('')
        pulls['comment'] = pulls.comment.apply(lambda x: x.split('Reviewed-on')[0])

        # comments part
        comments_file = data['comments_file']
        comments_file.time = pd.to_datetime(comments_file.time).dt.tz_localize(None)
        comments_file = comments_file[time_interval(comments_file.time, self.from_date, self.to_date)]
        comments_file = comments_file.drop(['index'], axis=1).rename({'time': 'date'}, axis=1)
        comments_file['key_file'] = comments_file['key_file'].apply(lambda x: x.replace(':', '/'))

        comments_pull = data['comments_patch']
        comments_pull.time = pd.to_datetime(comments_pull.time).dt.tz_localize(None)
        comments_pull = comments_pull[time_interval(comments_pull.time, self.from_date, self.to_date)]

        comments_pull = comments_pull.drop(['index', 'oid'], axis=1).rename({'time': 'date'}, axis=1)
        comments_pull['key_file'] = None

        comments = pd.concat((comments_pull, comments_file), axis=0).reset_index().drop(['index'], axis=1)

        return pulls, commits, comments

    def prepare_pulls(self):
        """
        pull preprocessing
        """

        # removes unnecessary columns and renames some of them
        self.pulls = self.pulls[
            ['file_path', 'key_change', 'reviewer_login', 'created_at', 'owner', 'comment', 'status',
             'updated_at']].rename(
            {'created_at': 'date', 'comment': 'title', 'updated_at': 'closed'}, axis=1)

        # group entries by pulls and aggregates reviewers
        self.pulls = self.pulls.groupby('key_change')[
            ['file_path', 'reviewer_login', 'date', 'owner', 'title', 'status', 'closed']].agg(
            {'file_path': lambda x: list(set(x)),
             'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0],
             'owner': lambda x: list(set(x)),
             'title': lambda x: list(x)[0],
             'status': lambda x: list(x)[0],
             'closed': lambda x: list(x)[0]}).reset_index()

        self.pulls['title'] = self.pulls['title'].fillna('')

        # get contributor for each of the pull from the commits and add them to the pulls
        pull_authors = self.commits.groupby('key_change').agg({'key_user': lambda x: set(x)}).reset_index()
        pull_authors = pull_authors.rename({'key_user': 'author'}, axis=1)
        self.pulls = self.pulls.merge(pull_authors, on='key_change', how='left')

    def from_checkpoint(self, path):
        self.pulls = pd.read_csv(path + '/pulls.csv', index_col=0)
        self.pulls.date = pd.to_datetime(self.pulls.date).dt.tz_localize(None)

        self.pulls.file_path = self.pulls.file_path.apply(ast.literal_eval)
        self.pulls.reviewer_login = self.pulls.reviewer_login.apply(ast.literal_eval)

        self.pulls.owner = self.pulls.owner.apply(lambda x: ast.literal_eval(x) if x is not np.nan else [])
        self.pulls.author = self.pulls.author.apply(lambda x: ast.literal_eval(x) if x is not np.nan else [])

        self.pulls = self.pulls.fillna('')

        try:
            self.pulls.reviewer_login = self.pulls.reviewer_login.apply(lambda x: [int(i) for i in x])
            self.pulls.author = self.pulls.author.apply(
                lambda x: [int(i) for i in x])
        except ValueError:
            pass

        self.commits = pd.read_csv(path + '/commits.csv', index_col=0)
        self.commits.date = pd.to_datetime(self.commits.date).dt.tz_localize(None)
        self.comments = pd.read_csv(path + '/comments.csv', index_col=0)
        self.comments.date = pd.to_datetime(self.comments.date).dt.tz_localize(None)

    def to_checkpoint(self, path):
        """
        saves dataset

        :param path: path to the folder to save results
        """

        if not os.path.exists(path):
            os.makedirs(path)

        self.pulls.to_csv(path + '/pulls.csv')
        self.commits.to_csv(path + '/commits.csv')
        self.comments.to_csv(path + '/comments.csv')
