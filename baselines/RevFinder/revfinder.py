from datetime import datetime, timedelta

import numpy as np

from baselines.RevFinder.utils import LCSubseq, LCSubstr, LCSuff, LCP
from baselines.Tie.utils import get_map


class RevFinder:
    def __init__(self, reviewer_list, max_date=100):
        self.history = []
        self.max_date = max_date

        self.reviewer_list = reviewer_list
        self.rev_count = len(reviewer_list)
        self.reviewer_map = get_map(reviewer_list)

        self._similarity_cache = [{} for _ in range(4)]

    def predict(self, review, n=10):
        metrics = [LCP, LCSuff, LCSubstr, LCSubseq]

        review = self._transform_review_format(review)

        rev_scores = [np.zeros(self.rev_count) for _ in metrics]

        end_time = review["date"]
        start_time = (datetime.fromtimestamp(end_time) - timedelta(days=self.max_date)).timestamp()

        cf1 = review["file_path"][:500]

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
            t = np.sum(rev_scores[metric_id] != 0)
            order_score[np.argsort(rev_scores[metric_id])] = np.arange(self.rev_count)
            final_score += np.maximum(order_score - np.sum(rev_scores[metric_id] == 0) + 1, 0)
        final_sorted_revs = np.argsort(final_score)

        return [self.reviewer_list[x] for x in final_sorted_revs[-n:][::-1]]

    def fit(self, review):
        self.history.append(self._transform_review_format(review))

    def _transform_review_format(self, review):
        reviewer_indices = [self.reviewer_map[_reviewer] for _reviewer in review["reviewer_login"]]
        return {
            "reviewer_login": np.array(reviewer_indices),
            "id": review["number"],
            "date": review["date"].timestamp(),
            "file_path": [f.split('/') for f in review["file_path"]]
        }
