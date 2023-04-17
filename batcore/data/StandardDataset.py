import os
from copy import deepcopy

import numpy as np
import pandas as pd

from batcore.bat_logging import Logger
from batcore.data.DatasetBase import DatasetBase
from batcore.data.utils import ItemMap, preprocess_users, add_self_review
import ast


class StandardDataset(DatasetBase, Logger):
    """
    dataset for most of the implemented models.

    :param dataset: GerritLoader-like object
    :param max_file: maximum number of files that a review can have
    :param commits: if False commits are omitted from the data
    :param comments: if False comments are omitted from the data
    :param user_items: if True user2id map is created
    :param file_items: if True file2id map is created
    :param pull_items: if true pull2id map is created
    :param owner_policy: how pull owners are calculated.
        * None - owners are unchanged
        * author - commit authors of the pull are treated as owners
        * author_no_na - commit authors of the pull are treated as owners. pulls without an author are removed
        * author_owner_fallback - if pull has author, owner field set to the author. Otherwise, nothing is done
    :param remove: list of columns to remove from the reviewers. Can be a subset of ['owner', 'author']
    :param factorize_users: when true users are replaced by id
    :param alias: True if clustering of the users by name should be performed
    :param bots: strategy for bot identification in user factorization. When 'auto' bots will be determined automatically. Otherwise, path to the csv with bot accounts should be specified
    :param project_name: name of the project for automatic bot detection
    :param self_review_flag: when true adds a column to the pulls dataframe which signifies that there was a self-review (based on the users aliases)
    """

    def __init__(self,
                 dataset=None,
                 max_file=86,
                 commits=False,
                 comments=False,
                 user_items=False,
                 file_items=False,
                 pull_items=False,
                 remove_empty=False,
                 owner_policy='author_owner_fallback',
                 remove='none',
                 process_users=False,
                 factorize_users=True,
                 alias=False,
                 remove_bots=True,
                 bots='auto',
                 project_name='',
                 self_review_flag=False,
                 checkpoint_path=None,
                 verbose=False,
                 log_file_path=None,
                 log_stdout=False,
                 log_mode='a'
                 ):

        self.setup_logger(verbose, log_file_path, log_stdout, log_mode)

        self.checkpoint = checkpoint_path is not None
        self.bad_pulls = None
        self.max_file = max_file
        self.commits = commits
        self.comments = comments

        if checkpoint_path is not None:
            self.info(f'loading from checkpoint {checkpoint_path}')
            self.from_checkpoint(checkpoint_path)
            self.info(f'finished loading from checkpoint {checkpoint_path}')
        if remove == 'none':
            remove = ['owner']

        dataset = deepcopy(dataset)
        if process_users:
            self.info(f'starting processing users')
            if self_review_flag:
                dataset_alias = deepcopy(dataset)
                preprocess_users(dataset_alias, remove_bots, bots, factorize_users, True, project_name, threshold=0.1)

            if self_review_flag and alias:
                dataset = dataset_alias
            else:
                preprocess_users(dataset, remove_bots, bots, factorize_users, alias, project_name, threshold=0.1)
            if self_review_flag:
                add_self_review(dataset_alias, dataset)
            self.info(f'finished processing users')

        self.user_items = user_items
        if self.user_items:
            self.users = None

        self.file_items = file_items
        if self.file_items:
            self.files = None

        self.pull_items = pull_items
        if self.pull_items:
            self.pulls = None

        self.owner_policy = owner_policy
        self.remove = remove
        self.remove_empty = remove_empty

        self.info(f'processing all data')
        super().__init__(dataset)

    def preprocess(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocess all necessary events and returns them as data stream
        """

        if not self.checkpoint:
            pulls = self.get_pulls(dataset)

            events = {'pulls': pulls}
            if self.commits:
                commits = self.get_commits(dataset)
                events['commits'] = commits

            if self.comments:
                comments = self.get_comments(dataset)
                events['comments'] = comments
        else:
            events = self.data

        self.additional_preprocessing(events)
        return events

    def get_pulls(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocessed pulls dataframe
        """
        # remove opened pulls. only Merged and Abandoned stay
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        # remember big pulls
        # self.bad_pulls = self.bad_pulls.union(set(pulls[pulls.file.apply(len) > self.max_file]['key_change']))
        self.bad_pulls = set(pulls[pulls.file.apply(len) > self.max_file]['key_change'])

        if self.remove_empty:
            # remember pull w/out reviewers
            self.bad_pulls = self.bad_pulls.union(set(pulls[pulls.reviewer.apply(len) == 0]['key_change']))

        # remove big pulls and pull w/out reviewers
        pulls = pulls[pulls.file.apply(len) <= self.max_file]
        if self.remove_empty:
            pulls = pulls[pulls.reviewer.apply(len) > 0]

        # owner estimation
        if self.owner_policy == 'author':
            pulls.owner = pulls.author
        elif self.owner_policy == 'author_no_na':
            pulls.owner = pulls.author
            pulls = pulls.dropna()
        elif self.owner_policy == 'author_owner_fallback':
            pulls.owner = pulls.apply(lambda x: list(x.author) if len(x.author) else (x.owner), axis=1)
        # elif self.owner_policy == 'none':
        #     pulls.owner = pulls.owner.apply(lambda x: [int(i) for i in x])
        else:
            raise ValueError(f'Wrong owner_policy {self.owner_policy}')

        # pulls.reviewer = pulls.reviewer.apply(lambda x: [int(i) for i in x])

        if len(self.remove):
            for col in self.remove:
                pulls.reviewer = pulls.apply(lambda x: [rev for rev in x['reviewer'] if rev not in x[col]],
                                             axis=1)

        # add label
        pulls.loc[:, 'type'] = 'pull'

        return pulls

    def get_commits(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocessed commits dataframe
        """
        commits = dataset.commits
        # remove commits to the bad pulls
        commits = commits[~commits['key_change'].isin(self.bad_pulls)]
        if commits.shape[0] > 0:
            commits.loc[:, 'type'] = 'commit'
        return commits

    def get_comments(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocessed commits dataframe
        """
        comments = dataset.comments
        # remove comments to the bad pulls
        comments = comments[~comments['key_change'].isin(self.bad_pulls)]
        # comments['type'] = 'comment'
        if comments.shape[0] > 0:
            comments.loc[:, 'type'] = 'comment'
        return comments

    def itemize_users(self, events):
        """
        creates user2id map from events
        """
        user_list = []
        if 'pulls' in events:
            pulls = events['pulls']
            user_list += pulls['reviewer'].sum() + pulls['owner'].sum()
        if 'comments' in events:
            user_list += events['comments']['key_user'].to_list()
        if 'commits' in events:
            user_list += events['commits']['key_user'].to_list()
        self.users = ItemMap(user_list)

        # if 'pulls' in events:
        # pulls = events['pulls']
        # pulls['reviewer'] = pulls['reviewer'].apply(lambda x: [self.users.getid(u) for u in x])
        # pulls['owner'] = pulls['owner'].apply(lambda x: [self.users.getid(u) for u in x])
        # if 'comments' in events:
        #     events['comments']['key_user'] = events['comments']['key_user'].apply(lambda x: self.users.getid(x))
        # if 'commits' in events:
        #     events['commits']['key_user'] = events['commits']['key_user'].apply(lambda x: self.users.getid(x))

    def itemize_pulls(self, events):
        """
        creates pull2id map from events
        """
        self.pulls = ItemMap(events['pulls']['key_change'])

    def itemize_files(self, events):
        """
        creates file2id map from events
        """
        self.files = ItemMap(events['pulls']['file'].sum())

    def additional_preprocessing(self, events):
        """
        creates all item2id maps
        """
        if self.user_items:
            self.itemize_users(events)
        if self.pull_items:
            self.itemize_pulls(events)
        if self.file_items:
            self.itemize_files(events)

    def replace(self, data, rev):
        data = deepcopy(data)
        l = len(data['reviewer_list'])
        data['reviewer_list'][np.random.randint(l)] = rev

        return data

    def get_items2ids(self):
        ret = {}
        if self.user_items:
            ret['users'] = self.users
        if self.pull_items:
            ret['pulls'] = self.pulls
        if self.file_items:
            ret['files'] = self.files
        return ret

    def from_checkpoint(self, path):

        self.data = {}
        self.data['pulls'] = pd.read_csv(path + '/pulls.csv', index_col=0)

        self.data['pulls'].date = pd.to_datetime(self.data['pulls'].date).dt.tz_localize(None)

        self.data['pulls'].file = self.data['pulls'].file.apply(ast.literal_eval)
        self.data['pulls'].reviewer = self.data['pulls'].reviewer.apply(ast.literal_eval)

        self.data['pulls'].owner = self.data['pulls'].owner.apply(
            lambda x: ast.literal_eval(x) if x is not np.nan else [])
        self.data['pulls'].author = self.data['pulls'].author.apply(
            lambda x: ast.literal_eval(x) if x is not np.nan else [])

        self.data['pulls'] = self.data['pulls'].fillna('')

        try:
            self.data['pulls'].reviewer = self.data['pulls'].reviewer.apply(lambda x: [int(i) for i in x])
            self.data['pulls'].author = self.data['pulls'].author.apply(lambda x: [int(i) for i in x])
        except ValueError:
            pass

        if self.commits:
            self.data['commits'] = pd.read_csv(path + '/commits.csv', index_col=0)
            self.data['commits'].date = pd.to_datetime(self.data['commits'].date).dt.tz_localize(None)

        if self.comments:
            self.data['comments'] = pd.read_csv(path + '/comments.csv', index_col=0)
            self.data['comments'].date = pd.to_datetime(self.data['comments'].date).dt.tz_localize(None)

    def to_checkpoint(self, path):
        """
        saves dataset

        :param path: path to the folder to save results
        """

        if not os.path.exists(path):
            os.makedirs(path)

        self.data['pulls'].to_csv(path + '/pulls.csv')
        self.data['commits'].to_csv(path + '/commits.csv')
        self.data['comments'].to_csv(path + '/comments.csv')
