import pandas as pd

from batcore.data.StandardDataset import StandardDataset
from .utils import ItemMap
from .utils import get_all_reviewers, get_all_words


class RevFinderDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.reviewers = get_all_reviewers(data)

    def get_items2ids(self):
        ret = {'reviewers': self.reviewers}
        return ret


class RevRecDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.users = ItemMap()
        for event in data:
            if event['type'] == 'pull':
                for rev in event['reviewer_login']:
                    self.users.add2(rev)
        for user in pd.unique(events['pulls']['owner'].sum()):
            self.users.add2(user)

        for user in pd.unique(events['comments']['key_user']):
            self.users.add2(user)
        #
        # if 'commits' in events:
        #     for user in pd.unique(events['commits']['key_user']):
        #         self.users.add2(user)


class TieDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.reviewers = get_all_reviewers(data)
        self.words = get_all_words(data)

    def get_items2ids(self):
        ret = {'reviewer_list': self.reviewers,
               'word_list': self.words}
        return ret
