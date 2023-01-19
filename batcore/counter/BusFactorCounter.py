from collections import defaultdict
from datetime import datetime

import numpy as np

from batcore.counter.CounterBase import CounterBase


class FileInfo:
    def __init__(self, author, creation_time, deletion_time=datetime(year=3000, month=1, day=1)):
        self.author = author
        self.creation_time = creation_time
        self.deletion_time = deletion_time


class BusFactorCounter(CounterBase):
    def __init__(self, data):
        self.files = defaultdict(lambda: None)
        self.commits = defaultdict(lambda: defaultdict(lambda: []))
        self.reviews = defaultdict(lambda: defaultdict(lambda: []))
        self.when_left = defaultdict(lambda: datetime(year=1900, month=1, day=1))

        self.data = data
        self.prepare()

    def prepare(self):
        for i, commit in self.data.commits.iterrows():
            file = commit.key_file
            user = commit.key_user
            time = commit.date

            self.commits[file][user].append(time)
            if self.files[file] is None:
                self.files[file] = FileInfo(user, time)

            self.when_left[user] = max(time, self.when_left[user])

        for i, review in self.data.pulls.iterrows():
            user = review.reviewer_login
            time = review.created_at
            self.when_left[user] = max(time, self.when_left[user])

    def reset(self):
        self.reviews = defaultdict(lambda: defaultdict(lambda: []))

    def count(self, history, from_date=None, to_date=None, S=220 * 60):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        for review in history:
            for file in review['file_path']:
                for user in review['reviewer_login']:
                    self.reviews[file][user].append(review['date'])

        DL = defaultdict(lambda: defaultdict(lambda: 0))
        RV = defaultdict(lambda: defaultdict(lambda: 0))

        DL_sum = defaultdict(lambda: 0)
        RV_sum = defaultdict(lambda: 0)

        DOA = defaultdict(lambda: defaultdict(lambda: 0))
        DOA_norm = defaultdict(lambda: 0)

        for file in self.commits:
            user_list = self.commits[file]
            for user in user_list:
                date_list = user_list[user]
                for date in date_list:
                    if to_date >= date:
                        time_delta = (to_date - date).seconds
                        DL[file][user] += np.exp(-time_delta / S)

        for file in self.reviews:
            user_list = self.reviews[file]
            for user in user_list:
                date_list = user_list[user]
                for date in date_list:
                    if to_date >= date:
                        time_delta = (to_date - date).seconds
                        RV[file][user] += np.exp(-time_delta / S)

        for file in DL:
            DL_sum[file] = sum(DL[file].values())

        for file in RV:
            RV_sum[file] = sum(RV[file].values())

        for file, file_info in self.files.items():
            DOA[file][file_info.author] += 3

            for user, user_dl in DL[file].items():
                DOA[file][user] += user_dl + 2.4 * np.log(1 + DL_sum[file]) - 2.4 * np.log(1 + DL_sum[file] - user_dl)

            for user, user_rv in RV[file].items():
                DOA[file][user] += user_rv + 2.4 * np.log(1 + RV_sum[file]) - 2.4 * np.log(1 + RV_sum[file] - user_rv)

        for file in DOA:
            DOA_norm[file] = max(DOA[file].values())

        user_knowledge = defaultdict(lambda: set())
        file_knowledge = defaultdict(lambda: 0)

        for file in DOA:
            norm_score = DOA_norm[file]
            for user in DOA[file]:
                if self.when_left[user] > to_date:
                    score = DOA[file][user]
                    if score >= 1 and score >= 0.75 * norm_score:
                        user_knowledge[user].add(file)
                        file_knowledge[file] += 1

        known_files = sum([1 for val in file_knowledge.values() if val != 0])
        sorted_users = sorted(user_knowledge.items(), key=lambda x: -len(x[1]))

        busfactor = 0
        while known_files * 2 > len(file_knowledge):
            print(busfactor)
            print(len(sorted_users))
            for file in sorted_users[busfactor][1]:
                fk = file_knowledge[file]
                if fk <= 0:
                    continue
                elif fk == 1:
                    known_files -= 1
                    file_knowledge[file] = 0
                else:
                    file_knowledge[file] = fk - 1

            busfactor += 1
            if busfactor == len(sorted_users):
                break

        self.reset()

        return busfactor

