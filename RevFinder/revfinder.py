from collections import defaultdict
from datetime import timedelta

import numpy as np

from RevFinder.utils import *
from RecommenderBase.recommender import RecommenderBase


class Review:
    def __init__(self, file_paths, reviewers, date):
        self.files = file_paths
        self.revs = reviewers
        self.date = date


class RevFinder(RecommenderBase):
    def __init__(self, max_date=100):
        super(RevFinder, self).__init__()
        self.history = []
        self.max_date = timedelta(max_date, 100)

    def predict_single_review(self, file_path, date, n=10):
        metrics = {'LCP': LCP,
                   'LCSuff': LCSuff,
                   'LCSubstr': LCSubstr,
                   'LCSubseq': LCSubseq}
        rev_scores = {metric: defaultdict(lambda: 0) for metric in metrics}

        for old_rev in reversed(self.history):
            if (date - old_rev.date) > self.max_date:
                break
            score = {metric: 0 for metric in metrics}
            for f1 in old_rev.files:
                for f2 in file_path:
                    for metric in metrics:
                        score[metric] += metrics[metric](f1, f2)

            for metric in metrics:
                if score[metric] > 0:
                    score[metric] /= len(file_path) * len(old_rev.files)
                    for rev in old_rev.revs:
                        rev_scores[metric][rev] += score[metric]

        final_score = defaultdict(lambda: 0)
        for metric in metrics:
            sorted_revs = [k for k, v in sorted(rev_scores[metric].items(), key=lambda item: item[1])]
            for i, rev in enumerate(sorted_revs):
                final_score[rev] += len(sorted_revs) - i

        final_sorted_revs = [k for k, v in sorted(final_score.items(), key=lambda item: item[1])]
        # if len(final_sorted_revs) < n:
        #     final_sorted_revs =  final_sorted_revs + [np.nan] * (n - len(final_sorted_revs))
        if len(final_sorted_revs) == 0:
            return [np.nan] * n
        return final_sorted_revs[:n]

    def predict(self, data, n=10):
        preds = []
        for _, row in data.iterrows():
            preds.append(self.predict_single_review(row.file_path, row.date, n))

        return preds

    def fit(self, data):
        for _, row in data.iterrows():
            self.history.append(Review(row.file_path, row.reviewer_login, row.date))


