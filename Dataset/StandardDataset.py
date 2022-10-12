import numpy as np

from Dataset.dataset import DatasetBase
from Dataset.utils import ItemMap


class StandardDataset(DatasetBase):
    def __init__(self,
                 dataset,
                 max_file=200,
                 commits=False,
                 comments=False,
                 user_items=False,
                 file_items=False,
                 pull_items=False,
                 owner_policy=None,
                 remove_owners=True
                 ):
        """
        :param max_file: maximum number of files that a review can have
        """
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
        self.remove_owners = remove_owners

        super().__init__(dataset)

    def preprocess(self, dataset):

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

        return data

    def get_pulls(self, dataset):
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        self.bad_pulls = set(pulls[pulls.reviewer_login.apply(len) == 0]['key_change'])
        self.bad_pulls = self.bad_pulls.union(set(pulls[pulls.file_path.apply(len) > self.max_file]['key_change']))

        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]
        if self.owner_policy == 'author':
            pulls.owner = pulls.author
        elif self.owner_policy == 'author_no_na':
            pulls.owner = pulls.author
            pulls = pulls.dropna()
        elif self.owner_policy == 'author_owner_fallback':
            pulls.owner = pulls.apply(lambda x: x.author if len(x.author) else x.owner, axis=1)
        else:
            pulls.owner = pulls.reviewer_login.apply(lambda x: [int(i) for i in x])

        pulls.reviewer_login = pulls.reviewer_login.apply(lambda x: [int(i) for i in x])
        if self.remove_owners:
            pulls.reviewer_login = pulls.apply(lambda x: [rev for rev in x['reviewer_login'] if rev not in x['owner']],
                                               axis=1)
            pulls = pulls[pulls.reviewer_login.apply(lambda x: len(x) > 0)]
        pulls['type'] = 'pull'

        return pulls

    def get_commits(self, dataset):
        commits = dataset.commits
        commits = commits[~commits['key_change'].isin(self.bad_pulls)]
        commits['type'] = 'commit'
        return commits

    def get_comments(self, dataset):
        comments = dataset.comments
        comments = comments[~comments['key_change'].isin(self.bad_pulls)]
        comments['type'] = 'comment'
        return comments

    def itemize_users(self, events):
        user_list = []
        if 'pulls' in events:
            pulls = events['pulls']
            user_list += pulls['reviewer_login'].sum() + pulls['owner'].to_list()
        if 'comments' in events:
            user_list += events['comments']['key_user'].to_list()
        if 'commits' in events:
            user_list += events['commits']['key_user'].to_list()
        self.users = ItemMap(user_list)

    def itemize_pulls(self, events):
        self.pulls = ItemMap(events['pulls']['key_change'])

    def itemize_files(self, events):
        self.files = ItemMap(events['pulls']['file_path'].sum())

    def additional_preprocessing(self, events, data):
        if self.user_items:
            self.itemize_users(events)
        if self.pull_items:
            self.itemize_pulls(events)
        if self.file_items:
            self.itemize_files(events)

    def replace(self, data, cur_rec):
        pass
