from collections import defaultdict
from datetime import datetime

from batcore.modelbase.recommender import BanRecommenderBase


class cHRev(BanRecommenderBase):
    """
    cHRev recommends candidates based on their commenting history. For this xFactor is calculated which measures
    relative portion and time recency of comments done by a candidate to the files

    Paper: `Automatically Recommending Peer Reviewers in Modern Code Review <https://ieeexplore.ieee.org/document/7328331>`_

    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)

        # reviewer expertise map. dict with (file, commenter)-triplet relation
        # triplet - (number of comments to the file, number of workdays when file was commented on, last comment date)
        self.re = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))

        # file review map - dict with file-triplet relation
        # triplet - (number of comments to the file, number of workdays when file was commented on, last comment date)
        self.fr = defaultdict(lambda: [0, 0, None])

        # for each file-user pair stores last comment date
        self.re_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))
        # for each file stores last comment date
        self.fr_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))

    def predict(self, pull, n=10):
        """
        scoring of candidates is performed by xFactor (equation 1-2 from the paper)

        :param pull: pull requests for which reviewers are required
        :param n: number of reviewers to recommend
        :return: at most n reviewers for the pull request
        """
        scores = defaultdict(lambda: 0)
        for file in pull['file']:
            file_val = self.fr[file]
            for user in self.re[file]:
                user_val = self.re[file][user]
                scores[user] += user_val[0] / file_val[0] + user_val[1] / file_val[1]
                if user_val[2] == file_val[2]:
                    scores[user] += 1
                else:
                    scores[user] += 1 / (user_val[2] - file_val[2]).days

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        return sorted_users[:n]

    def fit(self, data):
        super().fit(data)
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
