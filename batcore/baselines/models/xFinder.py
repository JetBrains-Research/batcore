from collections import defaultdict
from datetime import datetime

import numpy as np

from batcore.modelbase.recommender import BanRecommenderBase


class xFinder(BanRecommenderBase):
    """

    xFinder recommends candidates based on their committing history. For this xFactor is calculated which measures
    relative portion and time recency of commits done by a developer to the files

    dataset - StandardDataset(data, commits=True)

    Paper: `Assigning change requests to software developers <https://onlinelibrary.wiley.com/doi/abs/10.1002/smr.530>`_

    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)

        # developer-coder map. dict with keys (file, developer) and triplet values
        # triplets are - (number of commits to the file, number of days spending committing, date of most recent commit)
        # by the developer to the file
        self.dv = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))

        # file-code vector. dict with files as keys and triplet values
        # triplets are - (total number of commits to the file,
        #                 total number of days spending committing,
        #                 date of most recent commit)
        # done by any developer to the file
        self.fv = defaultdict(lambda: [0, 0, None])

        # maps that stores dates of most recent commits done to the file by any developer and by the specific developer
        self.dv_date = defaultdict(lambda: defaultdict(lambda: datetime(year=1000, month=1, day=1).date()))
        self.fc_date = defaultdict(lambda: datetime(year=1000, month=1, day=1).date())

        # additional scores for the case of recommendations to a single file. for pull requests they are not viable
        # self.package_score = defaultdict(lambda: defaultdict(lambda: 0))
        # self.project_score = defaultdict(lambda: defaultdict(lambda: 0))

    def predict(self, pull, n=10):
        """
        scoring of candidates is performed by xFactor (equations from page 8)

        :param pull: pull requests for which reviwers are required
        :param n: number of reviewers to recommend
        :return: at most n reviewers for the pull request
        """
        scores = defaultdict(lambda: 0)
        for file in pull['file']:
            file_val = self.fv[file]
            for user in self.dv[file]:
                user_val = self.dv[file][user]
                scores[user] += 1 / np.sqrt((file_val[0] - user_val[0]) ** 2 +
                                            (file_val[1] - user_val[1]) ** 2 +
                                            (file_val[2] - user_val[2]).days ** 2 + 1e-8)

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])

        # result = sorted_users[:n]

        return sorted_users[:n]

    def fit(self, data):
        super().fit(data)
        for event in data:
            if event['type'] == 'commit':
                # normal score
                file = event['key_file']
                user = event['key_user']
                date = event['date']

                val = self.dv[file][user]
                if self.dv_date[file][user] != date.date():
                    val[1] += 1
                val[0] += 1
                val[2] = date

                self.dv[file][user] = val

                val = self.fv[file]
                if self.fc_date[file] != date.date():
                    val[1] += 1

                val[0] += 1
                val[2] = date

                self.fv[file] = val

                self.dv_date[file][user] = date.date()
                self.fc_date[file] = date.date()

                # package score
                # package = '/'.join(file.split('/')[:-1])
                # self.package_score[package][user] += 1

                # project score
                # project = file.split('/', 1)[0]
                # self.project_score[project][user] += 1
