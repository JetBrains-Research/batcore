from copy import deepcopy

import numpy as np

from batcore.data.DatasetBase import DatasetBase
from batcore.data.utils import ItemMap


class StandardDataset(DatasetBase):
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
    """

    def __init__(self,
                 dataset,
                 max_file=100,
                 commits=False,
                 comments=False,
                 user_items=False,
                 file_items=False,
                 pull_items=False,
                 remove_empty=False,
                 owner_policy='author_owner_fallback',
                 remove='none'
                 ):

        if remove == 'none':
            remove = ['owner']

        self.bad_pulls = None
        self.max_file = max_file
        self.commits = commits
        self.comments = comments

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

        super().__init__(dataset)

    def preprocess(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocess all necessary events and returns them as data stream
        """

        pulls = self.get_pulls(dataset)

        events = {'pulls': pulls}
        if self.commits:
            commits = self.get_commits(dataset)
            events['commits'] = commits

        if self.comments:
            comments = self.get_comments(dataset)
            events['comments'] = comments
        data = []
        for event_type in events:
            data += events[event_type].to_dict('records')
        data = sorted(data, key=lambda x: x['date'])

        self.additional_preprocessing(events, data)
        self.events = events
        return data

    def get_pulls(self, dataset):
        """
        :param dataset: GerritLoader-like dataset
        :return: preprocessed pulls dataframe
        """
        # remove opened pulls. only Merged and Abandoned stay
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        # remember big pulls
        # self.bad_pulls = self.bad_pulls.union(set(pulls[pulls.file_path.apply(len) > self.max_file]['key_change']))
        self.bad_pulls = set(pulls[pulls.file_path.apply(len) > self.max_file]['key_change'])

        if self.remove_empty:
        # remember pull w/out reviewers
            self.bad_pulls = self.bad_pulls.union(set(pulls[pulls.reviewer_login.apply(len) == 0]['key_change']))

        # remove big pulls and pull w/out reviewers
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]
        if self.remove_empty:
            pulls = pulls[pulls.reviewer_login.apply(len) > 0]

        # owner estimation
        if self.owner_policy == 'author':
            pulls.owner = pulls.author
        elif self.owner_policy == 'author_no_na':
            pulls.owner = pulls.author
            pulls = pulls.dropna()
        elif self.owner_policy == 'author_owner_fallback':
            pulls.owner = pulls.apply(lambda x: x.author if len(x.author) else x.owner, axis=1)
        elif self.owner_policy == 'none':
            pulls.owner = pulls.owner.apply(lambda x: [int(i) for i in x])
        else:
            raise ValueError(f'Wrong owner_policy {self.owner_policy}')

        pulls.reviewer_login = pulls.reviewer_login.apply(lambda x: [int(i) for i in x])

        if len(self.remove):
            for col in self.remove:
                pulls.reviewer_login = pulls.apply(lambda x: [rev for rev in x['reviewer_login'] if rev not in x[col]],
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
        comments.loc[:, 'type'] = 'comment'
        return comments

    def itemize_users(self, events):
        """
        creates user2id map from events
        """
        user_list = []
        if 'pulls' in events:
            pulls = events['pulls']
            user_list += pulls['reviewer_login'].sum() + pulls['owner'].sum()
        if 'comments' in events:
            user_list += events['comments']['key_user'].to_list()
        if 'commits' in events:
            user_list += events['commits']['key_user'].to_list()
        self.users = ItemMap(user_list)

    def itemize_pulls(self, events):
        """
        creates pull2id map from events
        """
        self.pulls = ItemMap(events['pulls']['key_change'])

    def itemize_files(self, events):
        """
        creates file2id map from events
        """
        self.files = ItemMap(events['pulls']['file_path'].sum())

    def additional_preprocessing(self, events, data):
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
