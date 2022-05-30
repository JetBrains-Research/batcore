import numpy as np
import pandas as pd

from .recommender import RecommenderBase
from recsys.estimator import train_als
from recsys.mapping import MappingWithFallback, Mapping
from recsys.recommender import _recommend


class ALSRecommender(RecommenderBase):
    def __init__(self):
        super().__init__()
        self.commit_file_mapping = None
        self.pull_file_mapping = None
        self.commit_user_mapping = None
        self.pull_user_mapping = None

        self.model_commits_als = None
        self.model_reviews_als = None

        self.pulls = None
        self.commits = None

    def preprocess(self, data):
        pulls, commits = data
        id2file = np.unique(np.hstack((pulls.file_path, commits.file_path)))
        id2user = np.unique(np.hstack((pulls.reviewer_login, pulls.author_login, commits.author_login)))

        file2id = {f: i for i, f in enumerate(id2file)}
        user2id = {u: i for i, u in enumerate(id2user)}

        pulls['file_path'] = pulls.file_path.apply(lambda x: file2id[x])
        pulls['reviewer_login'] = pulls.reviewer_login.apply(lambda x: user2id[x])
        pulls['author_login'] = pulls.author_login.apply(lambda x: user2id[x])

        commits['file_path'] = commits.file_path.apply(lambda x: file2id[x])
        commits['author_login'] = commits.author_login.apply(lambda x: user2id[x])

        # TODO make one class with 2 masks
        self.commit_file_mapping = MappingWithFallback(file2id, id2file)
        self.pull_file_mapping = MappingWithFallback(file2id, id2file)

        self.commit_user_mapping = Mapping(file2id, id2file)
        self.pull_user_mapping = Mapping(file2id, id2file)

        return pulls, commits

    def predict(self, data, n=10):


        y_pred = []

        for i, (number, files, rev) in data.iterrows():
            _y_pred = _recommend(self.model_reviews_als,
                                 self.model_commits_als,
                                 self.pull_user_mapping,
                                 self.commit_user_mapping,
                                 self.pull_file_mapping,
                                 files,
                                 n)

            y_pred.append(_y_pred)

        return y_pred


    def fit(self, data):
        pulls, commits = data

        pulls = pulls.groupby(['file_path', 'reviewer_login', 'author_login']).count().reset_index().drop(
            ['date', 'author_login'], axis=1).rename({'reviewer_login': 'login'}, axis=1)
        if self.pulls is None:
            self.pulls = pulls
        else:
            self.pulls = pd.concat([self.pulls, pulls], axis=0).groupby(['file_path', 'login']).sum().reset_index()

        commits = commits.groupby(['file_path', 'author_login']).count().reset_index().rename(
            {'date': 'number', 'author_login': 'login'}, axis=1)
        if self.commits is None:
            self.commits = commits
        else:
            self.commits = pd.concat([self.commits, commits], axis=0).groupby(
                ['file_path', 'login']).sum().reset_index()

        self.model_commits_als, _ = train_als(self.commits)
        self.model_reviews_als, _ = train_als(self.pulls)

        self.commit_user_mapping.set_mask(self.commits, 'login')
        self.pull_user_mapping.set_mask(self.pulls, 'login')

        self.commit_file_mapping.set_mask(self.commits, 'file_path')
        self.pull_file_mapping.set_mask(self.pulls, 'file_path')
