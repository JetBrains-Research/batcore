from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix

from batcore.modelbase.recommender import BanRecommenderBase
from batcore.baselines.PRCF.utils import pearson, LCP


class PRCF(BanRecommenderBase):
    """
    dataset - comments, user_items, pull_items
    """
    def __init__(self,
                 users,
                 pulls,
                 lambd=0.7,
                 emd=20,
                 k=0.5, d=30,
                 delta=500,
                 freq=100,
                 lr=1e-3,
                 num_epochs=10,
                 lambd2=0.05,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)

        self.mat = dok_matrix((len(users), len(pulls)))
        self.com_cnt = dok_matrix((len(users), len(pulls)))

        self.history = []
        self.revs = set()

        self.users = users
        self.pulls = pulls

        self.begin_time = None
        self.end_time = None

        self.lambd = lambd
        self.k = k
        self.d = d
        self.delta = delta

        self.freq = freq
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambd2 = lambd2

        self.p = np.random.normal(size=(len(users), emd))
        self.q = np.random.normal(size=(len(pulls), emd))
        self.w = np.random.normal(size=(len(pulls), len(pulls)))

    def predict(self, pull, n=10):
        close_reviews = self.transform(pull)

        scores = defaultdict(lambda: 0)
        for user in self.revs:
            score = 0
            i = self.users.getid(user)
            for pull in close_reviews:
                j = self.pulls.getid(pull['key_change'])
                if self.mat[i, j] != 0:
                    score += self.mat[i, j] / (self.end_time - self.begin_time).seconds
                else:
                    score += self.implicit_eval(i, j)
            scores[user] = score

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        return sorted_users[:n]

    def fit(self, data):
        if self.no_inactive:
            self.update_time(data)
        for event in data:
            if self.begin_time is None:
                self.begin_time = event['date']
            if event['type'] == 'pull':
                self.history.append(event)
                self.revs.update(event['reviewer_login'])
            elif event['type'] == 'comment':

                pull = event['key_change']
                user = event['key_user']
                date = event['date']

                i = self.users.getid(user)
                j = self.pulls.getid(pull)

                self.mat[i, j] += pow(self.lambd, self.com_cnt[i, j]) * (date - self.begin_time).seconds
                self.com_cnt[i, j] += 1

        self.end_time = data[-1]['date']

        if (len(self.history) - 1) % self.freq == 0:
            self.train_emb()

    def transform(self, pull):
        result = []
        for pull_old in self.history[-self.delta:]:
            score = 0
            for f1 in pull['file_path']:
                for f2 in pull_old['file_path']:
                    score += LCP(f1, f2)
            score /= len(pull['file_path']) * len(pull_old['file_path'])
            if score > self.k:
                result.append(pull_old)
        return result

    def implicit_eval(self, i, j):
        cols = np.unique(self.mat.nonzero()[1])
        rd_ij = self.get_r(i, j, cols)
        mat_hat = (self.q[i] * self.p[i]).sum()
        neighborhood_score = 0
        den = (self.end_time - self.begin_time).seconds
        if len(rd_ij) != 0:
            for h in rd_ij:
                neighborhood_score += (self.mat[i, j] / den - mat_hat) * self.w[j, h]
            neighborhood_score /= np.sqrt(len(rd_ij))
        mat_hat += neighborhood_score
        return mat_hat

    def get_r(self, i, j, cols):
        scores = np.zeros(len(cols))
        col_j = self.mat.getcol(j)
        for i, h in enumerate(cols):
            if h == j:
                scores[i] = 1
            col_h = self.mat.getcol(h)
            n_jh = np.intersect1d(col_h.nonzero()[0], col_j.nonzero()[0]).shape[0]
            pearson_jh = pearson(col_j, col_h)
            scores[i] = n_jh / (100 + n_jh) * pearson_jh
        ind = np.argsort(scores)
        sd_j = cols[ind[1: self.d + 1]]
        r_i = self.mat.getrow(i).nonzero()[1]

        rd_ij = np.intersect1d(sd_j, r_i)

        return rd_ij

    def train_emb(self):
        cols = np.unique(self.mat.nonzero()[1])
        den = (self.end_time - self.begin_time).seconds
        for (i, j) in zip(*self.mat.nonzero()):
            rd_ij = self.get_r(i, j, cols)
            mat_hat = (self.q[i] * self.p[i]).sum()
            neighborhood_score = 0
            if len(rd_ij) != 0:
                for h in rd_ij:
                    neighborhood_score += (self.mat[i, j] / den - mat_hat) * self.w[j, h]
                neighborhood_score /= np.sqrt(len(rd_ij))
            mat_hat += neighborhood_score

            delta_pi = -2 * (self.mat[i, j] / den - mat_hat) * self.q[j] + 2 * self.lambd2 * self.p[i]
            delta_qj = -2 * (self.mat[i, j] / den - mat_hat) * self.p[i] + 2 * self.lambd2 * self.q[i]

            if len(rd_ij) != 0:
                t1 = (self.q[rd_ij] * self.w[j, rd_ij].reshape(-1, 1)).sum(axis=0)
                delta_pi += 2 * (self.mat[i, j] / den - mat_hat) * t1 / np.sqrt(len(rd_ij))

                t2 = (self.w[j, rd_ij].reshape(-1, 1) * self.p[i])
                delta_qh = 2 * (self.mat[i, j] / den - mat_hat) / np.sqrt(len(rd_ij)) * t2

                t3 = self.mat[i, rd_ij].toarray() / den - self.q[rd_ij] @ self.q[i]
                delta_wij = -2 * (self.mat[i, j] / den - mat_hat) * t3 / np.sqrt(len(rd_ij)) + 2 * self.w[j, rd_ij]

                self.w[j, rd_ij] -= self.lr * delta_wij
                self.q[rd_ij] -= self.lr * delta_qh

            self.p[i] -= self.lr * delta_pi
            self.q[j] -= self.lr * delta_qj
