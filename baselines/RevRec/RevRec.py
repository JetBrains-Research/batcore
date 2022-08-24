from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import dok_matrix

from RecommenderBase.recommender import RecommenderBase
from baselines.RevRec.utils import norm, sim


class RevRec(RecommenderBase):
    def __init__(self,
                 users,
                 k=0.5,
                 ga_params=None):
        super().__init__()

        if ga_params is None:
            self.ga_params = {'max_rev': 10,
                              'min_rev': 1,
                              'size': 20,
                              'prob': 0.1,
                              'max_eval': 100,
                              'k': 10,
                              'n': 10,
                              'alpha': 0.5,
                              'beta': 0.5}
        else:
            self.ga_params = ga_params

        self.k = k

        self.start_date = None
        self.end_date = None

        self.com_file = defaultdict(lambda: defaultdict(lambda: 0))
        self.com_date = defaultdict(lambda: defaultdict(lambda: None))

        self.users = users
        self.rc_graph = dok_matrix((len(users), len(users)))

        self.pull_file_part = defaultdict(lambda: defaultdict(lambda: set()))
        self.pull_owner = defaultdict(lambda: None)

        self.active_revs = 0
        self.test = set()

    def get_re_score(self, r, re):
        if self.ga_params['min_rev'] <= r.sum() <= self.ga_params['max_rev']:
            return (r * re).sum() / r.sum()
        return 0

    def get_rc_score(self, r, d):
        if not self.ga_params['min_rev'] <= r.sum() <= self.ga_params['max_rev']:
            return 0
        ind = np.arange(self.active_revs)[r]
        ind = np.hstack((ind, d))
        ind = np.unique(ind)

        sub_graph = self.rc_graph[ind][:, ind]

        v = sub_graph.shape[0]
        if v <= 1:
            return 0
        e = sub_graph.nonzero()[0].shape[0] / 2
        sum_e = sub_graph.sum() / 2
        return 2 * e / v / (v - 1) * sum_e

    def new_gen(self, ps, inds):
        children = []
        #         print(ps)
        for i in range(len(inds) // 2):
            cut = np.random.randint(len(ps[0]))
            child = np.zeros_like(ps[0])
            child[:cut] = ps[2 * i][:cut].copy()
            child[cut:] = ps[2 * i + 1][cut:].copy()

            mutation_mask = np.random.uniform(size=len(child)) > self.ga_params['prob']
            child[mutation_mask] = ~child[mutation_mask]
            children.append(child)
        return np.vstack(children)

    def get_scores(self, ps, re, d):
        re_scores = norm(np.array([self.get_re_score(p, re) for p in ps]))
        rc_scores = norm(np.array([self.get_rc_score(p, d) for p in ps]))
        return self.ga_params['alpha'] * re_scores + self.ga_params['beta'] * rc_scores

    def run_ga(self, d, re):
        ps = np.random.normal(size=(self.ga_params['size'], self.active_revs)) > 0.2
        #         print(ps)
        for _ in range(self.ga_params['max_eval']):
            scores = self.get_scores(ps, re, d)
            if scores.sum() != 0:
                parents_id = np.random.choice(np.arange(len(ps)), p=scores / scores.sum(), size=len(ps))
            else:
                parents_id = np.random.choice(np.arange(len(ps)), size=len(ps))

            children = self.new_gen(ps, parents_id)
            best_parents = ps[np.argsort(-scores)[:len(scores) // 2]]

            ps = np.vstack((best_parents, children))

        scores = self.get_scores(ps, re, d)
        ps = ps[np.argsort(-scores)]
        best = ps[0]
        cnt = ps[: self.ga_params['k']].sum(axis=0)
        return best, cnt

    def predict(self, review, n=10):
        scores_re = defaultdict(lambda: 0)

        cf = defaultdict(lambda: 0)
        cr = defaultdict(lambda: datetime(year=10, month=1, day=1))

        for file in review['file_path']:
            for f2 in self.com_file:
                if sim(file, f2) > self.k:
                    for user in self.com_file[f2]:
                        cf[user] += self.com_file[f2][user]
                        cr[user] = max(cr[user], self.com_date[f2][user])
        for user in cf:
            scores_re[user] = cf[user] * (1 - (self.end_date - cr[user]).days / (self.end_date - self.start_date).days)

        re = np.zeros(self.active_revs)
        for i in range(self.active_revs):
            re[i] = scores_re[self.users[i]]

        best, cnt = self.run_ga(self.users.getid(review['owner']), re)

        best_id = np.arange(len(best))[best]
        cnt = cnt[best]

        recs = [self.users[best_id[i]] for i in np.argsort(-cnt)]
        return recs

    def fit(self, data):
        if self.start_date is None:
            self.start_date = data[0]['date']

        for event in data:
            if event['type'] == 'pull':
                pull = event['key_change']
                user = event['owner']
                self.pull_owner[pull] = self.users.getid(user)

                for rev in event['reviewer_login']:
                    self.active_revs = max(self.active_revs, self.users.getid(rev) + 1)
                    self.test.add(self.users.getid(rev))

            elif event['type'] == 'comment':
                pull = event['key_change']
                owner_id = self.pull_owner[pull]
                user = event['key_user']
                user_id = self.users.getid(user)

                if 'key_file' in event:

                    file = event['key_file']

                    self.com_file[file][user] += 1
                    self.com_date[file][user] = event['date']

                    for i in self.pull_file_part[pull][file]:
                        if i == user_id:
                            continue
                        self.rc_graph[user_id, i] += 1
                        self.rc_graph[i, user_id] += 1
                    self.pull_file_part[pull][file].add(user_id)

                if (owner_id is not None) and (user_id != owner_id):
                    self.rc_graph[owner_id, user_id] += 1
                    self.rc_graph[user_id, owner_id] += 1

        self.end_date = data[-1]['date']