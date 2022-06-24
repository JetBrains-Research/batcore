from collections import defaultdict

import numpy as np

from RecommenderBase.recommender import RecommenderBase


class TextMiningRecommender(RecommenderBase):
    def __init__(self, dataset):
        super().__init__()

        all_revs = list(set(dataset.pulls.reviewer_login.sum()))

        self.id2rev = all_revs
        self.rev2id = {r: i for i, r in enumerate(all_revs)}

        self.cond_prob = np.zeros((len(self.id2rev), len(dataset.pulls.body.iloc[0])))
        self.den = np.zeros(len(self.id2rev))

        self.prob = np.zeros(len(self.id2rev))
        self.rev_count = 0

        self.cur_rev = np.zeros(len(self.id2rev), dtype=bool)

    def predict_single_review(self, row, n=None):

        log_prob = np.log(self.cond_prob + 1e-8) - np.log(self.den + 1e-8).reshape(-1, 1)
        log_prob = log_prob * row.body
        log_prob = log_prob.sum(-1)
        log_prob += np.log(self.prob + 1e-8)

        scores = defaultdict(lambda: 0, {self.id2rev[i]: np.exp(log_prob[i]) for i in range(len(log_prob))})

        if n is None:
            return scores
        else:
            final_sorted = [k for k, v in sorted(scores.items(), key=lambda item: -item[1])]

            if len(final_sorted) == 0:
                return [np.nan] * n
            return final_sorted[:n]

    def set_new_ids(self):
        self.new_ids = np.arange(len(self.cur_rev))[self.cur_rev]

    def predict(self, data, n=10):
        self.set_new_ids()
        res = []

        for i, row in data.iterrows():
            res.append(self.predict_single_review(row, n))
        return res

    def fit(self, data):
        for (i, row) in data.iterrows():
            rev_id = np.array([self.rev2id[rev] for rev in row.reviewer_login])

            self.prob[rev_id] += 1
            self.rev_count += 1

            self.cond_prob[rev_id] += row.body
            self.den[rev_id] += row.body.sum()

            self.cur_rev[rev_id] = 1
