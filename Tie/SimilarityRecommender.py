from collections import defaultdict
from datetime import timedelta

import numpy as np

from RecommenderBase.recommender import RecommenderBase
from RevFinder.revfinder import Review
from Tie.utils import revsim


class SimilarityRecommender(RecommenderBase):
    def __init__(self, max_date=100):
        super().__init__()
        self.history = []
        self.max_date = timedelta(max_date, 100)

    def predict_single_review(self, file_path, date, n=10):
        rev_scores = defaultdict(lambda: 0)

        for old_rev in reversed(self.history):
            if (date - old_rev.date) > self.max_date:
                break
            score = 0
            for f1 in old_rev.files:
                for f2 in file_path:
                    score += revsim(f1, f2)

            if score > 0:
                score /= len(file_path) * len(old_rev.files)
                for rev in old_rev.revs:
                    rev_scores[rev] += score

        final_sorted = [k for k, v in sorted(rev_scores.items(), key=lambda item: item[1])]

        if len(final_sorted) == 0:
            return [np.nan] * n
        return final_sorted[:n]

    def predict(self, data, n=10):
        preds = []
        for _, row in data.iterrows():
            preds.append(self.predict_single_review(row.file_path, row.date, n))

        return preds

    def fit(self, data):
        for _, row in data.iterrows():
            self.history.append(Review(row.file_path, row.reviewer_login, row.date))
