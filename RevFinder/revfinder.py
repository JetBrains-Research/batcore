import multiprocessing as mp
from collections import defaultdict
from datetime import timedelta
from itertools import product

# import billiard as mp
import numpy as np

from RecommenderBase.recommender import RecommenderBase
from RevFinder.utils import *


class Review:
    def __init__(self, file_paths, reviewers, date):
        self.files = file_paths
        self.revs = reviewers
        self.date = date


class ArgIterator:
    def __init__(self, iterator, *args):
        self.it = iterator
        self.args = args
        self.iter_it = None

    def __iter__(self):
        self.iter_it = iter(self.it)
        return self

    def __next__(self):
        res = next(self.iter_it)
        return res, self.args


class RevFinder(RecommenderBase):
    def __init__(self, dataset, max_date=100):
        super(RevFinder, self).__init__()
        self.history = []
        self.max_date = timedelta(max_date, 100)

        self.log = defaultdict(lambda: [])

        self.hash = -np.ones((4, len(dataset.id2file), len(dataset.id2file)))
        self.cnt = defaultdict(lambda: 0)

        self.file2id = dataset.file2id
        self.id2file = dataset.id2file

        self.pools = mp.Pool(10)

    def predict_single_review(self, file_path, date, n=10):
        metrics = {
            'LCP': LCP,
            'LCSuff': LCSuff,
            'LCSubstr': LCSubstr,
            'LCSubseq': LCSubseq
        }
        rev_scores = [defaultdict(lambda: 0) for metric in metrics]

        for metric_id, metric in enumerate(metrics):

            for old_rev in reversed(self.history):
                if (date - old_rev.date) > self.max_date:
                    break
                # results = [metrics[metric]((f1, f2)) for (f1, f2) in product(old_rev.files, file_path)]
                results = self.pools.map(metrics[metric], product(old_rev.files, file_path))
                score = sum(results)
                if score > 0:
                    score /= len(file_path) * len(old_rev.files)
                for rev in old_rev.revs:
                    rev_scores[metric_id][rev] += score

        final_score = defaultdict(lambda: 0)
        for (metric_id, metric) in enumerate(metrics):
            sorted_revs = [k for k, v in sorted(rev_scores[metric_id].items(), key=lambda item: item[1])]
            for i, rev in enumerate(sorted_revs):
                final_score[rev] += len(sorted_revs) - i

        final_sorted_revs = [k for k, v in sorted(final_score.items(), key=lambda item: item[1])]

        if len(final_sorted_revs) == 0:
            return [np.nan] * n
        return final_sorted_revs[:n]

    def predict(self, data, n=10):
        preds = []
        for _, row in data.iterrows():
            preds.append(self.predict_single_review(row.file_path, row.date, n))
        return preds

    def start_pool(self):
        self.pools = mp.Pool(10)

    def stop_pool(self):
        self.pools.close()

    def fit(self, data):
        for _, row in data.iterrows():
            self.history.append(Review(row.file_path, row.reviewer_login, row.date))
