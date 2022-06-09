import numpy as np

from RecommenderBase.recommender import RecommenderBase


class TextRecommender(RecommenderBase):
    def __init__(self, dataset):
        super().__init__()

        all_revs = list(set(dataset.pulls.reviewer_login.sum()))
        all_tokens = list(set(dataset.pulls.body.sum()))

        self.id2rev = all_revs
        self.rev2id = {r: i for i, r in enumerate(all_revs)}

        self.id2tok = all_tokens
        self.tok2id = {r: i for i, r in enumerate(all_tokens)}

        self.cond_prob = np.zeros((len(self.id2rev), len(self.id2tok)))
        self.prob = np.zeros(len(self.id2rev))
        self.den = np.zeros(len(self.id2rev))

        self.cur_rev = np.zeros(len(self.id2rev), dtype=bool)

    def predict_single_review(self, row):
        if len(row.body) == 0:
            probs = np.log(self.prob[self.cur_rev] + 1e-8)
        else:
            tok_id = np.array([self.tok2id[tok] for tok in row.body if tok in self.tok2id])
            tok_id, counts = np.unique(tok_id, return_counts=True)
            probs = self.cond_prob[self.cur_rev, :]
            probs = probs[:, tok_id] + 1e-8
            probs = np.log(probs)
            probs = probs * counts
            probs = probs.sum(axis=-1)
            probs += np.log(self.prob[self.cur_rev] + 1e-8) - len(row.body) * np.log(self.den[self.cur_rev] + 1e-8)
        best = np.argsort(probs)[:10]

        return best

    def predict(self, data, n=10):
        new_ids = np.arange(len(self.cur_rev))[self.cur_rev]
        res = []
        for i, row in data.iterrows():
            best = self.predict_single_review(row)
            best = new_ids[best][: n]
            best = [self.id2rev[i] for i in best]
            res.append(best)

        return res

    def fit(self, data):
        for (i, row) in data.iterrows():
            rev_id = np.array([self.rev2id[rev] for rev in row.reviewer_login])
            self.prob[rev_id] += 1
            if len(row.body):
                tok_id = np.array([self.tok2id[tok] for tok in row.body])
                tok_id, counts = np.unique(tok_id, return_counts=True)
                self.cond_prob[np.ix_(rev_id, tok_id)] += counts
                self.den[rev_id] += counts.sum()

            self.cur_rev[np.array([self.rev2id[rev] for rev in list(set(data.reviewer_login.sum()))])] = 1
