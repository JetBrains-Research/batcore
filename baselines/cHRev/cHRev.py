from collections import defaultdict
from datetime import datetime

from RecommenderBase.recommender import RecommenderBase


class cHRev(RecommenderBase):
    def __init__(self):
        super().__init__()

        self.re = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
        self.fr = defaultdict(lambda: [0, 0, None])

        self.re_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))
        self.fr_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))

    def predict(self, review, n=10):
        scores = defaultdict(lambda: 0)
        for file in review['file_path']:
            file_val = self.fr[file]
            for user in self.re[file]:
                user_val = self.re[file][user]
                scores[user] += user_val[0] / file_val[0] + user_val[1] / file_val[1]
                if user_val[2] == file_val[2]:
                    scores[user] += 1
                else:
                    scores[user] += 1 / (user_val[2] - file_val[2]).days

        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])


        return sorted_users[:n]

    def fit(self, data):
        for event in data:
            if event['type'] == 'comment':
                file = event['key_file']
                user = event['key_user']
                date = event['date']

                val = self.re[file][user]
                if self.re_date[file][user] != date.date():
                    val[1] += 1
                val[0] += 1
                val[2] = date

                self.re[file][user] = val

                val = self.fr[file]
                if self.fr_date[file] != date.date():
                    val[1] += 1

                val[0] += 1
                val[2] = date

                self.fr[file] = val

                self.re_date[file][user] = date.date()
                self.fr_date[file] = date.date()