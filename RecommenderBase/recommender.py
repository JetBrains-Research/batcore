from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta, datetime


class RecommenderBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, data, n=10):
        """
        predicts best n reviewers for each pull in Dataset
        :param data: Dataset for which reviewers need to be predicted
        :param n: number of reviewers to predict
        :return: list of recommendations
        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        trains RecommenderBase on given Dataset
        :param data: train Dataset to fit
        """
        pass


class BanRecommenderBase(RecommenderBase, ABC):
    def __init__(self, no_owner=True, no_inactive=True, inactive_time=60):
        super().__init__()
        self.no_owner = no_owner
        self.no_inactive = no_inactive
        self.inactive_time = timedelta(days=inactive_time)
        self.last_active = defaultdict(lambda: datetime(year=1950, day=1, month=1))

    @staticmethod
    def remove_user(scores, user_id):
        if user_id in scores:
            del scores[user_id]

    def remove_inactive(self, scores, cur_date):
        users_to_remove = []
        for user in scores:
            if (cur_date - self.last_active[user]) > self.inactive_time:
                users_to_remove.append(user)
        for user in users_to_remove:
            del scores[user]

    def update_time(self, events):
        for event in events:
            if event['type'] == 'pull':
                date = event['date']
                try:
                    self.last_active[event['owner']] = date
                except KeyError:
                    pass
                for reviewer in event['reviewer_login']:
                    self.last_active[reviewer] = date
            else:
                date = event['date']
                user = event['key_user']
                self.last_active[user] = date

    def filter(self, scores, pull):
        if self.no_owner:
            self.remove_user(scores, pull['owner'])
        if self.no_inactive:
            self.remove_inactive(scores, pull['date'])

    def fit(self, data):
        if self.no_inactive:
            self.update_time(data)

