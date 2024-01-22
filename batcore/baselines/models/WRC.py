import ast
import numpy as np
from functools import partial
from batcore.modelbase.recommender import BanRecommenderBase
from ..utils import LCP, path2list
from ray.util.multiprocessing import Pool
import ray


def count_score(f1, pull=None, wrc=None, files=None, users=None):
    results = np.zeros(wrc.shape[1])
    for f2 in pull['file']:
        for i in range(wrc.shape[1]):
            val = wrc[files.getid(f2), i]
            if val >= 0:
                results[i] += val * LCP(path2list(f1), path2list(f2))
    return results


class WRC(BanRecommenderBase):
    """
    WRC recommends reviewers based on the file path similarity measures. Files contribute to the final score not only
    based on their similary but also on their recency and size of their pull request.

    dataset - StandardDataset(data, user_items=True, file_items=True)

    Paper: `Automatically Recommending Code Reviewers Based on Their Expertise: An Empirical Comparison <https://dl.acm.org/doi/abs/10.1145/2970276.2970306>`_

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
        self.p = Pool(10)
        self.pool_cnt = 0
        self.lcp_results = -np.ones((len(self.files), len(self.files)))
        self.log = []

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
        scores = np.zeros(len(self.users))

        # for f1 in self.known_files:
        #     for f2 in pull['file_path']:
        #         for i in range(self.wrc.shape[1]):
        #             val = self.wrc[self.files.getid(f2), i]
        #             if val >= 0:
        #                 scores[i] += val * LCP(path2list(f1), path2list(f2))  # self.LCP_count(f1, f2)

        # f = time.time()
        # self.log.append(f-s)
        # params = product(self.known_files, pull['file_path'])
        # res = [count_score2(f1, f2, self.wrc, self.files) for f1, f2 in params]
        # res = [count_score2(f1, f2, self.wrc, self.files) for f1, f2 in params]

        # s = time.time()
        # res = self.p.starmap(partial(count_score2, wrc=self.wrc, files=self.files),
        #                      product(self.known_files, pull['file_path']))
        #

        res = self.p.map(partial(count_score, pull=pull, wrc=self.wrc, files=self.files, users=self.users),
                         self.known_files)
        # # f = time.time()
        # # print(f - s)
        res = np.vstack(res)
        scores = res.sum(axis=0)

        self.pool_cnt += 1
        if self.pool_cnt == 5:
            ray.shutdown()
            ray.init()
            self.p = Pool(10)
            self.pool_cnt = 0

        # params = product(self.known_files, pull['file_path'], range(self.wrc.shape[1]))
        # for (_), r in zip(params, res):
        #     for i, v in zip(range(self.wrc.shape[1]), r):
        #         scores[i] += v

        scores = {self.users[i]: scores[i] for i in range(len(self.users))}
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

                for file in event['file']:
                    self.known_files.add(file)
                    for user in event['reviewer']:
                        self.wrc[self.files.getid(file), self.users.getid(user)] += 1 / len(event['reviewer'])

    # def save(self, path='checkpoints'):
    #     with open(f"{path}/wrc/wrc.npy", 'wb') as f:
    #         np.save(f, self.wrc)
    #     with open(f"{path}/wrc/lcp.npy", 'wb') as f:
    #         np.save(f, self.lcp_results)
    #     with open(f"{path}/wrc/files.npy", 'w') as f:
    #         f.write(str(self.known_files))
    #
    # def load(self, path='checkpoints/wrc'):
    #     with open(f"{path}/wrc.npy", 'rb') as f:
    #         self.wrc = np.load(f)
    #     with open(f"{path}/lcp.npy", 'rb') as f:
    #         self.lcp_results = np.load(f)
    #     with open(f"{path}/files.npy", 'r') as f:
    #         self.known_files = ast.literal_eval(f.read())
