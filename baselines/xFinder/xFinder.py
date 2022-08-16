from collections import defaultdict
from datetime import datetime

import numpy as np

from RecommenderBase.recommender import RecommenderBase


class xFinder(RecommenderBase):
    def __init__(self):
        super().__init__()

        self.devcodemap = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
        self.filechange = defaultdict(lambda: [0, 0, None])

        self.devcode_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))
        self.file_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))

        self.package_score = defaultdict(lambda: defaultdict(lambda: 0))  # won't hold for reviews
        self.project_score = defaultdict(lambda: defaultdict(lambda: 0))  # will hold for reviews

    def predict(self, review, n=10):
        scores = defaultdict(lambda: 0)
        for file in review['file_path']:
            file_val = self.filechange[file]
            for user in self.devcodemap[file]:
                user_val = self.devcodemap[file][user]
                scores[user] += 1 / np.sqrt((file_val[0] - user_val[0]) ** 2 +
                                            (file_val[1] - user_val[1]) ** 2 +
                                            (file_val[2] - user_val[2]).days ** 2)

        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        # result = sorted_users[:n]

        return sorted_users[:n]

    def fit(self, data):
        for event in data:
            if event['type'] == 'commit':
                # normal score
                file = event['key_file']
                user = event['key_user']
                date = event['date']

                val = self.devcodemap[file][user]
                if self.devcode_date[file][user] != date.date():
                    val[1] += 1
                val[0] += 1
                val[2] = date

                self.devcodemap[file][user] = val

                val = self.filechange[file]
                if self.file_date[file] != date.date():
                    val[1] += 1

                val[0] += 1
                val[2] = date

                self.filechange[file] = val

                self.devcode_date[file][user] = date.date()
                self.file_date[file] = date.date()

                # package score
                package = '/'.join(file.split('/')[:-1])
                self.package_score[package][user] += 1

                # project score
                project = file.split('/', 1)[0]
                self.project_score[project][user] += 1
