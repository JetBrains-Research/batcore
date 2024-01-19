from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import dok_matrix

from batcore.modelbase.recommender import BanRecommenderBase
from ..utils import norm, sim


class RevRec(BanRecommenderBase):
    """
    RevRec finds best set of reviewers based on the two metrics: group expertise on modified files and amount of
    collaborations with pull request submitter. The search for the best set is performed via genetic algorithm

    Paper: `Search-Based Peer Reviewers Recommendation in Modern Code Review <https://ieeexplore.ieee.org/document/7816482>`_

    :param items2ids: dict with users2ids
    :param k: threshold for files similarity
    :param ga_params: dict of hyperparameters for genetic algorithm

        :ga_parameter max_rev: maximum number of reviewers to recommend. default=10
        :ga_parameter min_rev: minimum number of reviewers to recommend. default=1
        :ga_parameter size: population size. default=20
        :ga_parameter prob: mutation probability. default=0.1
        :ga_parameter max_eval: number of genetic algorithm iterations. default=100
        :ga_parameter n: number of best solutions that contribute to the sorting of the best reviewers. default=10
        :ga_parameter alpha: weight of the reviewer expertise score. default=0.5
        :ga_parameter beta: weight of the reviewer collaboration score. default=0.5
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 items2ids,
                 k=0.5,
                 ga_params=None,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)

        if ga_params is None:
            self.ga_params = {'max_rev': 10,
                              'min_rev': 1,
                              'size': 20,
                              'prob': 0.1,
                              'max_eval': 100,
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

        self.users = items2ids['users']
        self.rc_graph = dok_matrix((len(self.users), len(self.users)))

        self.pull_file_part = defaultdict(lambda: defaultdict(lambda: set()))
        self.pull_owners = defaultdict(lambda: [])

        self.active_revs = 0
        # self.test = set()

        self.banned = None

    def get_re_score(self, candidate, expertise):
        """
        counts expertise score
        """
        if (candidate * self.banned).sum() > 0:
            return 0
        if self.ga_params['min_rev'] <= candidate.sum() <= self.ga_params['max_rev']:
            return (candidate * expertise).sum() / candidate.sum()
        return 0

    def get_rc_score(self, candidate, owners):
        """
        counts collaboration score
        """
        if (candidate * self.banned).sum() > 0:
            return 0
        if not self.ga_params['min_rev'] <= candidate.sum() <= self.ga_params['max_rev']:
            return 0
        ind = np.arange(self.active_revs)[candidate]
        ind = np.hstack((ind, owners))
        ind = np.unique(ind)

        sub_graph = self.rc_graph[ind][:, ind]

        v = sub_graph.shape[0]
        if v <= 1:
            return 0
        e = sub_graph.nonzero()[0].shape[0] / 2
        sum_e = sub_graph.sum() / 2
        return 2 * e / v / (v - 1) * sum_e

    def new_gen(self, population, parents_ids):
        """
        :param population: current population
        :param parents_ids: ids of current candidates that will be produce next generation
        :return: new generation
        """
        children = []
        for i in range(len(parents_ids) // 2):
            cut = np.random.randint(len(population[0]))
            child = np.zeros_like(population[0])
            child[:cut] = population[2 * i][:cut].copy()
            child[cut:] = population[2 * i + 1][cut:].copy()

            mutation_mask = np.random.uniform(size=len(child)) > self.ga_params['prob']
            child[mutation_mask] = ~child[mutation_mask]
            children.append(child)
        return np.vstack(children)

    def get_scores(self, population, expertise, owners):
        """
        :return: for each candidate in :param population: counts expertise and collaboration score with weights
        """
        re_scores = norm(np.array([self.get_re_score(p, expertise) for p in population]))
        rc_scores = norm(np.array([self.get_rc_score(p, owners) for p in population]))
        return self.ga_params['alpha'] * re_scores + self.ga_params['beta'] * rc_scores

    def run_ga(self, owners, expertise):
        """
        runs a genetic algorithm to get reviewers recommendations

        :param owners: owners of the pull requests for which are recommendations are made
        :param expertise: expertise of the potential candidates
        :return: best set of reviewers and list of occurrences of each reviewer in the last population
        """
        population = np.random.normal(size=(self.ga_params['size'], self.active_revs)) > 0.2

        for _ in range(self.ga_params['max_eval']):
            scores = self.get_scores(population, expertise, owners)
            if scores.sum() != 0:
                parents_id = np.random.choice(np.arange(len(population)), p=scores / scores.sum(), size=len(population))
            else:
                parents_id = np.random.choice(np.arange(len(population)), size=len(population))

            children = self.new_gen(population, parents_id)
            best_parents = population[np.argsort(-scores)[:len(scores) // 2]]

            population = np.vstack((best_parents, children))

        scores = self.get_scores(population, expertise, owners)
        population = population[np.argsort(-scores)]
        best = population[0]
        cnt = population[: self.ga_params['n']].sum(axis=0)
        return best, cnt

    def set_banned(self, pull):
        """
        sets self.banned to a binary mask of candidates that won't be recommended
        """
        self.banned = np.zeros(self.active_revs)
        if self.no_owner:
            for owner in pull['owner']:
                owner_id = self.users.getid(owner)
                if owner_id < self.active_revs:
                    self.banned[owner_id] = 1
        if self.no_inactive:
            cur_date = pull['date']
            for user in self.last_active:
                if self.users.getid(user) < self.active_revs:
                    if (cur_date - self.last_active[user]) > self.inactive_time:
                        self.banned[self.users.getid(user)] = 1

    def predict(self, pull, n=10):
        """
        :param pull: pull requests for which reviwers are required
        :param n: number of reviewers to recommend
        :return: at most n reviewers for the pull request
        """

        # calculation of the candidate expertise based on the file path similarities of the changed files with
        # previously commented files

        if self.active_revs == 0:
            return []
        scores_re = defaultdict(lambda: 0)

        cf = defaultdict(lambda: 0)
        cr = defaultdict(lambda: datetime(year=10, month=1, day=1))

        for file in pull['file']:
            for f2 in self.com_file:
                if f2 is not None and sim(file, f2) > self.k:
                    for user in self.com_file[f2]:
                        cf[user] += self.com_file[f2][user]
                        cr[user] = max(cr[user], self.com_date[f2][user])

        for user in cf:
            scores_re[user] = cf[user] * (1 - ((self.end_date - cr[user]).seconds/86400) / ((self.end_date - self.start_date).seconds/ 86400))

        expertise = np.zeros(self.active_revs)
        for i in range(self.active_revs):
            expertise[i] = scores_re[self.users[i]]

        # setting banned reviewers
        self.set_banned(pull)

        owners_id = [self.users.getid(owner) for owner in pull['owner']]

        # running genetic algorithm
        best, cnt = self.run_ga(owners_id, expertise)
        best_id = np.arange(len(best))[best]
        cnt = cnt[best]

        recs = [self.users[best_id[i]] for i in np.argsort(-cnt)]
        return recs

    def fit(self, data):
        """
        builds a collaboration graph, and remembers comment interactions between developers and files
        """
        super().fit(data)
        if self.start_date is None:
            self.start_date = data[0]['date']

        for event in data:
            if event['type'] == 'pull':
                pull = event['key_change']
                self.pull_owners[pull] = [self.users.getid(user) for user in event['owner']]

                for rev in event['reviewer']:
                    self.active_revs = max(self.active_revs, self.users.getid(rev) + 1)

            elif event['type'] == 'comment':
                pull = event['key_change']
                owners_id = self.pull_owners[pull]
                user = event['key_user']
                user_id = self.users.getid(user)

                if 'key_file' in event:

                    file = event['key_file']
                    if file != file:
                        continue
                    self.com_file[file][user] += 1
                    self.com_date[file][user] = event['date']

                    for i in self.pull_file_part[pull][file]:
                        if i == user_id:
                            continue
                        self.rc_graph[user_id, i] += 1
                        self.rc_graph[i, user_id] += 1
                    self.pull_file_part[pull][file].add(user_id)

                for owner_id in owners_id:
                    if user_id != owner_id:
                        self.rc_graph[owner_id, user_id] += 1
                        self.rc_graph[user_id, owner_id] += 1

        self.end_date = data[-1]['date']

    def update_time(self, events):
        """
        for all the participants in each event updates time of most recent action
        :param events: batch of events
        """
        for event in events:
            if event['type'] == 'pull':
                date = event['date']
                try:
                    for owner in event['owner']:
                        self.last_active[owner] = date
                except KeyError:
                    pass
                for reviewer in event['reviewer']:
                    self.last_active[reviewer] = date
            elif event['type'] == 'comment':
                date = event['date']
                user = event['key_user']
                self.last_active[user] = date
