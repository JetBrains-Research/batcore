from collections import defaultdict

import numpy as np

from RevFinder.utils import *
from env.recommender import RecommenderBase


class Review:
    def __init__(self, file_paths, reviewers):
        self.files = file_paths
        self.revs = reviewers


class RevFinder(RecommenderBase):
    def __init__(self):
        super(RevFinder, self).__init__()
        self.history = []

    def preprocess(self, data):
        """
        Retain only pull data and group same pulls together
        """
        return [data[0].groupby('number')[['file_path', 'reviewer_login', 'date']].agg(
            {'file_path': list, 'reviewer_login': lambda x: list(set(x)), 'date': lambda x: list(x)[0]}).reset_index()]

    def predict_single_review(self, file_path, n=10):
        metrics = {'LCP': LCP,
                   'LCSuff': LCSuff,
                   'LCSubstr': LCSubstr,
                   'LCSubseq': LCSubseq}
        rev_scores = {metric: defaultdict(lambda: 0) for metric in metrics}

        for old_rev in self.history:
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
            preds.append(self.predict_single_review(row.file_path, n))

        return preds

    def fit(self, data):
        data = data[0]
        self.history = []
        for _, row in data.iterrows():
            self.history.append(Review(row.file_path, row.reviewer_login))


