from collections import defaultdict
from datetime import timedelta

from RecommenderBase.recommender import RecommenderBase


class ACRec(RecommenderBase):
    def __init__(self, gamma=60, lambd=0.5):
        super().__init__()

        self.gamma = timedelta(days=gamma)
        self.lambd = lambd

        self.history = []
        self.commenters = defaultdict(lambda: [])

    def predict(self, review, n=10):
        scores = defaultdict(lambda: 0)
        date = review['date']
        for review_old in reversed(self.history):
            if (date - review_old['date']) > self.gamma:
                break

            for com in self.commenters[review_old['key_change']]:
                scores[com] += pow((date - review_old['date']).days + 0.01, -self.lambd)

        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        for event in data:
            if event['type'] == 'pull':
                self.history.append(event)
            elif event['type'] == 'comment':
                self.commenters[event['key_change']].append(event['key_user'])