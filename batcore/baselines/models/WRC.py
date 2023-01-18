import ast
from collections import defaultdict
import numpy as np

from batcore.RecommenderBase.recommender import BanRecommenderBase
from ..utils import LCP, path2list


def count_score(f1, f2, i, wrc=None, files=None):
    val = wrc[files.getid(f2), i]
    if val >= 0:
        return val * LCP(path2list(f1), path2list(f2))
    return 0


class WRC(BanRecommenderBase):
    """
    WRC recommends reviewers based on the file path similarity measures. Files contribute to the final score not only
    based on their similary but also on their recency and size of their pull request.

    dataset - StandardDataset(data, user_items=True, file_items=True)

    Paper : "Automatically Recommending Code Reviewers Based on Their Expertise: An Empirical Comparison"

    :param items2ids: dict with user2id and file2id
    :param delta: time decay factor for weight of the previous reviews
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 items2ids,
                 delta=1,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)

        self.files = items2ids['files']
        self.users = items2ids['users']

        self.delta = delta
        self.reviews = []

        # matrix of wrc scores for (user, files) pairs
        self.wrc = np.zeros((len(self.files), len(self.users)))

        self.known_files = set()

        # self.scores = {}
        # self.p = Pool(5)
        self.lcp_results = -np.ones((len(self.files), len(self.files)))

    def LCP_count(self, f1, f2):
        f1_id = self.files.getid(f1)
        f2_id = self.files.getid(f2)

        if self.lcp_results[f1_id, f2_id] == -1:
            self.lcp_results[f1_id, f2_id] = LCP(path2list(f1), path2list(f2))

        return self.lcp_results[f1_id, f2_id]

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
                        scores[self.users[i]] += val * self.LCP_count(f1, f2)  # LCP(path2list(f1), path2list(f2))

        # params = product(self.known_files, pull['file_path'], range(self.wrc.shape[1]))
        # res = [count_score(f1, f2, i, self.wrc, self.files) for f1, f2, i in params]

        # res = self.p.starmap(partial(count_score, wrc=self.wrc, files=self.files),
        #                      product(self.known_files, pull['file_path'], range(self.wrc.shape[1])))
        #
        # params = product(self.known_files, pull['file_path'], range(self.wrc.shape[1]))
        # for (f1, f2, i), r in zip(params, res):
        #     scores[self.users[i]] += r

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

    def save(self, path='checkpoints'):
        with open(f"{path}/wrc/wrc.npy", 'wb') as f:
            np.save(f, self.wrc)
        with open(f"{path}/wrc/lcp.npy", 'wb') as f:
            np.save(f, self.lcp_results)
        with open(f"{path}/wrc/files.npy", 'w') as f:
            f.write(str(self.known_files))

    def load(self, path='checkpoints/wrc'):
        with open(f"{path}/wrc.npy", 'rb') as f:
            self.wrc = np.load(f)
        with open(f"{path}/lcp.npy", 'rb') as f:
            self.lcp_results = np.load(f)
        with open(f"{path}/files.npy", 'r') as f:
            self.known_files = ast.literal_eval(f.read())

