import queue
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix
import networkx as nx
from community import community_louvain

from RecommenderBase.recommender import RecommenderBase, BanRecommenderBase


class CN(BanRecommenderBase):
    """
    dataset - comments=True, user_items=True
    """
    def __init__(self, users, lambd=0.5,no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)

        self.w = dok_matrix((len(users), len(users)))
        self.com_cnt = defaultdict(lambda: defaultdict(lambda: 0))
        self.pull_owner = defaultdict(lambda: None)

        self.lambd = lambd

        self.users = users

        self.start_time = None
        self.end_time = None

        self.pull_count = 0
        self.num_revs = np.zeros(len(users))
        self.cooc = np.zeros((len(users), len(users)))

        self.in_deg = dok_matrix((len(users), 1))

    def predict(self, pull, n=10):

        owner = self.users.getid(pull['owner'])

        if len(self.w.getcol(owner).nonzero()[0]) > 0:
            recs = self.predict_pac(owner, n)
        elif len(self.w.getcol(owner).nonzero()[0]) > 0:
            recs = self.predict_apriori(owner, n)
        else:
            recs = self.predict_community(owner, n)

        recs = [self.users[i] for i in recs]
        return recs

    def fit(self, data):
        super().fit(data)
        new_end_time = data[-1]['date']
        for event in data:
            if event['type'] == 'pull':
                if self.start_time is None:
                    self.start_time = event['date']
                owner = event['owner']
                pull = event['key_change']
                self.pull_owner[pull] = self.users.getid(owner)

                self.pull_count += 1
                if len(event['reviewer_login']):
                    rev_ind = np.array([self.users.getid(rev) for rev in event['reviewer_login']])
                    self.num_revs += 1
                    self.cooc[tuple(np.meshgrid(rev_ind, rev_ind))] += 1

            elif event['type'] == 'comment':

                pull = event['key_change']
                user = event['key_user']
                date = event['date']

                i = self.pull_owner[pull]
                if i is None:
                    continue

                j = self.users.getid(user)
                if i == j:
                    continue

                val = pow(self.lambd, self.com_cnt[pull][j]) * (date - self.start_time).seconds
                old_den = (self.end_time - self.start_time).seconds
                new_den = (new_end_time - self.start_time).seconds
                self.w[i, j] = (self.w[i, j] * old_den + val) / new_den
                self.in_deg[j] = (self.in_deg[j] * old_den + val) / new_den

                self.com_cnt[pull][j] += 1

        self.end_time = new_end_time

    def predict_pac(self, i, k=10):
        mark = np.zeros(self.w.shape[0])
        q = queue.Queue()
        q.put(i)
        recs = []
        while not q.empty():
            if len(recs) >= k:
                break
            v = q.get()
            neighbours = self.w.getcol(v).nonzero()[1]
            neighbours = neighbours[mark[neighbours] == 0]
            scores = self.w[v, neighbours].toarray()
            for j in np.argsort(-scores)[0]:
                v_nb = neighbours[j]
                mark[v_nb] = 1
                recs.append(v_nb)
                q.put(v_nb)

                if len(recs) >= k:
                    break
        return recs

    def predict_apriori(self, i, k=10):
        scores = self.cooc[i]
        scores[i] = -1

        best_recs = np.argpartition(-scores, k)[:k]
        sorted_recs = sorted(best_recs, key=lambda x: -self.num_revs[x])

        return sorted_recs

    def predict_community(self, i, k=10):
        g = nx.Graph(self.w.toarray())
        cm = community_louvain.best_partition(g)

        com_nodes = defaultdict(lambda: [])
        for (node, com) in cm.items():
            com_nodes[com].append(node)

        com_size = {com: len(com_nodes[com]) for com in com_nodes}
        com_id, com_size = np.split(np.array(list(com_size.items())), 2, axis=1)

        # remove small
        com_id = com_id[com_size > 1]
        com_size = com_size[com_size > 1]

        # sort by size
        ind = np.argsort(-com_size)
        com_id = com_id[ind]
        com_size = com_size[ind]

        for com in com_id:
            com_nodes[com] = sorted(com_nodes[com], key=lambda x: -self.in_deg[x])

        recs = []
        empty_cnt = 0

        ind = 0
        while len(recs) < k and empty_cnt < len(com_id):
            for c, cs in zip(com_id, com_size):
                if ind >= cs:
                    continue
                if com_nodes[c][ind] != i:
                    recs.append(com_nodes[c][ind])
                if ind == cs - 1:
                    empty_cnt += 1
                if len(recs) >= k:
                    break
            ind += 1
        return recs