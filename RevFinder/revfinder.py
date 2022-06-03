from collections import defaultdict

from RevFinder.utils import LCP
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
        rev_scores = defaultdict(lambda: 0)

        for old_rev in self.history:
            score = 0
            for f1 in old_rev.files:
                for f2 in file_path:
                    score += LCP(f1, f2)

            if score > 0:
                score /= len(file_path) * len(old_rev.files)
                for rev in old_rev.revs:
                    rev_scores[rev] += score

        sorted_revs = [k for k, v in sorted(rev_scores.items(), key=lambda item: item[1])]
        return sorted_revs[:n]

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


