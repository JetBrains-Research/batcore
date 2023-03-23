from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import timedelta, datetime


class RecommenderBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, pull, n=10):
        """
        predicts best n reviewers for the pull

        :param pull: pull for which reviewers need to be predicted
        :param int n: number of reviewers to predict
        :return: list of recommendations
        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        trains RecommenderBase on given data
        :param data: train data to fit
        """
        pass


class BanRecommenderBase(RecommenderBase, ABC):
    """
    Base class for the recommender models with built-in filtering of the candidates.

    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """
    def __init__(self, no_owner=True, no_inactive=True, inactive_time=60):
        super().__init__()
        self.no_owner = no_owner
        self.no_inactive = no_inactive
        self.inactive_time = timedelta(days=inactive_time)
        self.last_active = defaultdict(lambda: datetime(year=1950, day=1, month=1))

    @staticmethod
    def remove_user(scores, user_id):
        """
        :param scores: dict with scores for each reviewer candidate
        :param user_id: id of candidate to be removed
        """
        if user_id in scores:
            del scores[user_id]

    def remove_inactive(self, scores, cur_date):
        """
        removes recently inactive users from the resulstin scores

        :param scores: dict with scores for each reviewer candidate
        :param cur_date: date of the pull request for which scores were calculated
        """
        users_to_remove = []
        for user in scores:
            if (cur_date - self.last_active[user]) > self.inactive_time:
                users_to_remove.append(user)
        for user in users_to_remove:
            del scores[user]

    def update_time(self, events):
        """
        for all the participants in each event updates time of most recent action

        :param events: batch of events
        """
        for event in events:
            if event['type'] == 'pull':
                date = event['date']
                try:
                    for owner in event['owner']:
                        self.last_active[owner] = date
                except KeyError:
                    pass
                for reviewer in event['reviewer']:
                    self.last_active[reviewer] = date
            else:
                date = event['date']
                user = event['key_user']
                self.last_active[user] = date

    def filter(self, scores, pull):
        """
        For the given pull request filters candidates

        :param pull: pull request for which recommendations are calculated
        :param scores: dict scores of potential candidates for the code review
        """
        if self.no_owner:
            for owner in pull['owner']:
                self.remove_user(scores, owner)
        if self.no_inactive:
            self.remove_inactive(scores, pull['date'])

    def fit(self, data):
        """
        performs necessary updates
        """
        if self.no_inactive:
            self.update_time(data)
