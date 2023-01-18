import queue
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix
import networkx as nx
from community import community_louvain

from batcore.RecommenderBase.recommender import BanRecommenderBase


# TODO look into multiple owners
class CN(BanRecommenderBase):
    """
    CN recommends reviewers based on their comments on previous reviews. For this a Comment Network (weighted directed graph is
    constructed). Vertices are developers in the project and edges represents weighted number of reviewing
    interactions. Scores to each candidate are assigned based on the distance in graph

    dataset - StandardDataset(data, comments=True, user_items=True)

    Paper: "Reviewer Recommendation for Pull-Requests in GitHub: What Can We Learn from Code Review and Bug Assignment?"

    :param items2ids: dict with user2id
    :param lambd: weight decay coefficient for comments within a single review beyond the first one
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 items2ids,
                 lambd=0.5,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)

        self.users = items2ids['users']

        # Adjacency matrix of developers comment interactions
        self.w = dok_matrix((len(self.users), len(self.users)))

        self.com_cnt = defaultdict(lambda: defaultdict(lambda: 0))
        self.pull_owner = defaultdict(lambda: None)

        self.lambd = lambd

        self.start_time = None
        self.end_time = None

        # total number of pull requests
        self.pull_count = 0
        # number of reviews performed by user
        self.num_revs = np.zeros(len(self.users))
        # matrix of co-occurrences of reviewers within pull-requests
        self.cooc = np.zeros((len(self.users), len(self.users)))

        self.in_deg = dok_matrix((len(self.users), 1))

    def predict(self, pull, n=10):
        """
        recommends reviewers based on owner of the pull request
        """

        owner = self.users.getid(pull['owner'][0])

        if len(self.w.getcol(owner).nonzero()[0]) > 0:
            recs = self.predict_pac(owner, n)
        elif len(self.w.getcol(owner).nonzero()[0]) > 0:
            recs = self.predict_apriori(owner, n)
        else:
            recs = self.predict_community(owner, n)

        recs = [self.users[i] for i in recs]
        return recs

    def fit(self, data):
        """
        updates CN graph and supporting characteristics
        """
        super().fit(data)
        new_end_time = data[-1]['date']
        for event in data:
            if event['type'] == 'pull':
                if self.start_time is None:
                    self.start_time = event['date']
                owner = event['owner'][0]
                pull = event['key_change']
                self.pull_owner[pull] = self.users.getid(owner)

                self.pull_count += 1
                if len(event['reviewer_login']):
                    rev_ind = np.array([self.users.getid(rev) for rev in event['reviewer_login']])
                    self.num_revs[rev_ind] += 1
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
                new_den = (new_end_time - self.start_time).seconds
                if self.in_deg[j] == 0:
                    self.w[i, j] = val / new_den
                    self.in_deg[j] = val / new_den
                else:
                    old_den = (self.end_time - self.start_time).seconds
                    self.w[i, j] = (self.w[i, j] * old_den + val) / new_den
                    self.in_deg[j] = (self.in_deg[j] * old_den + val) / new_den

                self.com_cnt[pull][j] += 1

        self.end_time = new_end_time

    def predict_pac(self, i, k=10):
        """
        PAC prediction recommendations are for the owners that have that had previous pull requests that have been reviewed e.i.
        their internal degree > 0
        """
        mark = np.zeros(self.w.shape[0])
        mark[i] = 1
        q = queue.Queue()
        q.put(i)
        recs = []
        while not q.empty():
            if len(recs) >= k:
                break
            v = q.get()
            neighbours_1 = self.w.getcol(v).nonzero()[0]
            neighbours = neighbours_1[mark[neighbours_1] == 0]
            scores = self.w[v, neighbours].toarray()
            for j in np.argsort(-scores)[0]:
                v_nb = neighbours[j]
                mark[v_nb] = 1
                if v_nb not in recs:
                    recs.append(v_nb)
                else:
                    raise NameError('pac is wrong')
                q.put(v_nb)

                if len(recs) >= k:
                    break
        return recs

    def predict_apriori(self, i, k=10):
        """
        Predict apriori suggest reviewers for the prs with owners that had commented previously on other prs
        """
        scores = self.cooc[i]
        scores[i] = -1

        best_recs = np.argpartition(-scores, k)[:k]
        sorted_recs = sorted(best_recs, key=lambda x: -self.num_revs[x])
        if len(sorted_recs) != len(np.unique(sorted_recs)):
            print('apriori')
        return sorted_recs

    def predict_community(self, i, k=10):
        """
        Suggest reviewers for prs of newcomers that had no interactions with others
        """
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
        if len(recs) != len(np.unique(recs)):
            print('com')
        return recs
