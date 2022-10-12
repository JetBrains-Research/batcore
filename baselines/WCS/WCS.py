from collections import defaultdict

import numpy as np

from RecommenderBase.recommender import RecommenderBase, BanRecommenderBase
from baselines.WCS.utils import LCP


class WCS(BanRecommenderBase):
    """
    dataset - user_items, file_items
    """
    def __init__(self, users, files, delta=1, no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)

        self.files = files
        self.users = users

        self.delta = delta
        self.reviews = []

        self.wcs = np.zeros((len(files), len(users)))

        self.known_files = set()

    def predict(self, pull, n=10):
        scores = defaultdict(lambda: 0)

        for f1 in self.known_files:
            for f2 in pull['file_path']:
                for i in range(self.wcs.shape[1]):
                    val = self.wcs[self.files.getid(f2), i]
                    if val >= 0:
                        scores[self.users[i]] += val * LCP(f1, f2)

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        return sorted_users[:n]

    def fit(self, events):
        if self.no_inactive:
            self.update_time(events)
        for review in events:
            self.reviews.append(review)
            self.wcs = self.delta * self.wcs

            for file in review['file_path']:
                self.known_files.add(file)
                for user in review['reviewer_login']:
                    self.wcs[self.files.getid(file), self.users.getid(user)] += 1 / len(review['reviewer_login'])
