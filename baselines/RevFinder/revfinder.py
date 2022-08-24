from datetime import datetime, timedelta

import numpy as np

from RecommenderBase.recommender import BanRecommenderBase
from baselines.RevFinder.utils import LCSubseq, LCSubstr, LCSuff, LCP
from baselines.Tie.utils import get_map


class RevFinder(BanRecommenderBase):
    def __init__(self, reviewer_list, max_date=100,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)
        self.history = []
        self.max_date = max_date

        self.reviewer_list = reviewer_list
        self.rev_count = len(reviewer_list)
        self.reviewer_map = get_map(reviewer_list)

        self._similarity_cache = [{} for _ in range(4)]

    def predict(self, pull, n=10):
        metrics = [LCP, LCSuff, LCSubstr, LCSubseq]

        pull = self._transform_review_format(pull)

        rev_scores = [np.zeros(self.rev_count) for _ in metrics]

        end_time = pull["date"]
        start_time = end_time - timedelta(days=self.max_date)

        cf1 = pull["file_path"][:500]

        order_score = np.zeros(self.rev_count)

        for old_rev in reversed(self.history):
            if old_rev['date'] == end_time:
                continue
            if old_rev['date'] < start_time:
                break

            cf2 = old_rev["file_path"][:500]

            for metric_id, metric in enumerate(metrics):
                score = 0
                for f1 in cf1:
                    for f2 in cf2:
                        score += metric(f1, f2)
                if score > 0:
                    score /= len(cf1) * len(cf2)

                for rev in old_rev['reviewer_login']:
                    rev_scores[metric_id][rev] += score

        final_score = np.zeros(self.rev_count)
        for (metric_id, metric) in enumerate(metrics):
            t = np.sum(rev_scores[metric_id] == 0)
            order_score[np.argsort(rev_scores[metric_id])] = np.arange(self.rev_count)
            final_score += np.maximum(order_score - t + 1, 0)

        scores = {self.reviewer_list[i]:s for i, s in enumerate(final_score)}
        self.filter(scores, pull)
        # final_sorted_revs = np.argsort(final_score)
        # return [self.reviewer_list[x] for x in final_sorted_revs[-n:][::-1]]

        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        super().fit(data)
        pull = data[0]
        self.history.append(self._transform_review_format(pull))

    def _transform_review_format(self, pull):
        reviewer_indices = [self.reviewer_map[_reviewer] for _reviewer in pull["reviewer_login"]]
        return {
            "reviewer_login": np.array(reviewer_indices),
            "id": pull["number"],
            "date": pull["date"],
            "file_path": [f.split('/') for f in pull["file_path"]],
            "owner": pull['owner']
        }
