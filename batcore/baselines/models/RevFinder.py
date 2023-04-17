import copy
from datetime import timedelta

import numpy as np

from batcore.modelbase.recommender import BanRecommenderBase
from ..utils import LCSubseq, LCSubstr, LCSuff, LCP
from ..utils import get_map


# TODO update
class RevFinder(BanRecommenderBase):
    """
    RevFinder suggest possible reviewers based on their previous reviews of the similar files. In RevFinder there are
    4 different file similarities metrics. For each metric list of suggestions is calculated, and then they are
    combined into one

    Paper: `Who Should Review My Code? A File Location-Based Code-Reviewer Recommendation Approach for Modern Code Review <https://ieeexplore.ieee.org/document/7081824>`_

    :param items2ids: dict with all possible reviewers
    :param max_date: time in days after which old reviews stop influence predictions
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 items2ids,
                 max_date=100,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)
        self.history = []
        self.max_date = max_date

        self.reviewer_list = items2ids['reviewers']
        self.rev_count = len(self.reviewer_list)
        self.reviewer_map = get_map(self.reviewer_list)

        # self._similarity_cache = [{} for _ in range(4)]

    def predict(self, pull, n=10):
        """
        :param n: number of reviewers to recommend
        """
        metrics = [LCP, LCSuff, LCSubstr, LCSubseq]

        pull = self.update_pull(pull)

        rev_scores = [np.zeros(self.rev_count) for _ in metrics]

        end_time = pull["date"]
        start_time = end_time - timedelta(days=self.max_date)

        cf1 = pull["file"][:500]

        order_score = np.zeros(self.rev_count)

        for old_rev in reversed(self.history):
            if old_rev['date'] == end_time:
                continue
            if old_rev['date'] < start_time:
                break

            cf2 = old_rev["file"][:500]

            for metric_id, metric in enumerate(metrics):
                score = 0
                for f1 in cf1:
                    for f2 in cf2:
                        score += metric(f1, f2)
                if score > 0:
                    score /= len(cf1) * len(cf2)

                for rev in old_rev['reviewer']:
                    rev_scores[metric_id][rev] += score

        final_score = np.zeros(self.rev_count)
        for (metric_id, metric) in enumerate(metrics):
            t = np.sum(rev_scores[metric_id] == 0)
            order_score[np.argsort(rev_scores[metric_id])] = np.arange(self.rev_count)
            final_score += np.maximum(order_score - t + 1, 0)

        scores = {self.reviewer_list[i]: s for i, s in enumerate(final_score)}
        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        """
        adds reviews into a history buffer
        """
        super().fit(data)
        for event in data:
            if event['type'] == 'pull':
                self.history.append(self.update_pull(event))

    def update_pull(self, pull):
        pull = copy.deepcopy(pull)

        reviewer_indices = [self.reviewer_map[_reviewer] for _reviewer in pull["reviewer"]]

        pull['reviewer'] = np.array(reviewer_indices)
        pull['file'] = [f.split('/') for f in pull["file"]]

        return pull
