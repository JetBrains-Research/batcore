from collections import defaultdict

import numpy as np

from RecommenderBase.recommender import BanRecommenderBase
from ..utils import LCP, path2list


class WRC(BanRecommenderBase):
    """
    WRC recommends reviewers based on the file path similarity measures. Files contribute to the final score not only
    based on their similary but also on their recency and size of their pull request.

    dataset - StandardDataset(data, user_items=True, file_items=True)

    Paper : "Automatically Recommending Code Reviewers Based on Their Expertise: An Empirical Comparison"
    """

    def __init__(self,
                 items2ids,
                 delta=1,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        """
        :param items2ids: dict with user2id and file2id
        :param delta: time decay factor for weight of the previous reviews
        :param no_owner: flag to add or remove owners of the pull request from the recommendations
        :param no_inactive: flag to add or remove inactive reviewers from recommendations
        :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
        """
        super().__init__(no_owner, no_inactive, inactive_time)

        self.files = items2ids['files']
        self.users = items2ids['users']

        self.delta = delta
        self.reviews = []

        # matrix of wrc scores for (user, files) pairs
        self.wrc = np.zeros((len(self.files), len(self.users)))

        self.known_files = set()

    def predict(self, pull, n=10):
        """
        counts sum of all wrc score for each possible reviewer and files in the pull
        :param n: number of reviewers to recommend
        """
        scores = defaultdict(lambda: 0)

        for f1 in self.known_files:
            for f2 in pull['file_path']:
                for i in range(self.wrc.shape[1]):
                    val = self.wrc[self.files.getid(f2), i]
                    if val >= 0:
                        scores[self.users[i]] += val * LCP(path2list(f1), path2list(f2))

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        return sorted_users[:n]

    def fit(self, data):
        """
        updates wrc matrix
        """
        if self.no_inactive:
            self.update_time(data)
        for event in data:
            if event['type'] == 'pull':
                self.reviews.append(event)
                self.wrc = self.delta * self.wrc

                for file in event['file_path']:
                    self.known_files.add(file)
                    for user in event['reviewer_login']:
                        self.wrc[self.files.getid(file), self.users.getid(user)] += 1 / len(event['reviewer_login'])
