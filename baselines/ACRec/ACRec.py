from collections import defaultdict
from datetime import timedelta

from RecommenderBase.recommender import BanRecommenderBase


class ACRec(BanRecommenderBase):
    def __init__(self, gamma=60, lambd=0.5,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)

        self.gamma = timedelta(days=gamma)
        self.lambd = lambd

        self.history = []
        self.commenters = defaultdict(lambda: [])

    def predict(self, pull, n=10):
        scores = defaultdict(lambda: 0)
        date = pull['date']
        for review_old in reversed(self.history):
            if (date - review_old['date']) > self.gamma:
                break

            for com in self.commenters[review_old['key_change']]:
                scores[com] += pow((date - review_old['date']).days + 0.01, -self.lambd)

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        super().fit(data)
        for event in data:
            if event['type'] == 'pull':
                self.history.append(event)
            elif event['type'] == 'comment':
                self.commenters[event['key_change']].append(event['key_user'])
