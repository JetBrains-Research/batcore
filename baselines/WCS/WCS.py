from collections import defaultdict

import numpy as np

from RecommenderBase.recommender import RecommenderBase
from baselines.WCS.utils import LCP


class WCS(RecommenderBase):
    def __init__(self, files, users, delta=1):
        super().__init__()

        self.files = files
        self.users = users

        self.delta = delta
        self.reviews = []

        self.wcs = np.zeros((len(files), len(users)))

        self.known_files = set()

    def predict(self, review, n=10):
        scores = defaultdict(lambda: 0)

        for f1 in self.known_files:
            for f2 in review['file_path']:
                for i in range(self.wcs.shape[1]):
                    val = self.wcs[self.files.getid(f2), i]
                    if val >= 0:
                        scores[self.users[i]] += val * LCP(f1, f2)

        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        return sorted_users[:n]

    def fit(self, review):
        self.reviews.append(review)
        self.wcs = self.delta * self.wcs

        for file in review['file_path']:
            self.known_files.add(file)
            for user in review['reviewer_login']:
                self.wcs[self.files.getid(file), self.users.getid(user)] += 1 / len(review['reviewer_login'])
