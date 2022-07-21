from collections import defaultdict

from Counter.CounterBase import CounterBase


class FaRCounter(CounterBase):
    def __init__(self, data):
        self.when_known = defaultdict(lambda: {})
        self.when_left = defaultdict(lambda: None)
        self.when_created = defaultdict(lambda: None)
        self.data = data
        self.prepare()
        self.wkr = []

    def prepare(self):

        for i, review in self.data.pulls.iterrows():
            file = review.file_path
            reviewer = review.reviewer_login

            if reviewer not in self.when_left:
                self.when_left[reviewer] = review['created_at']
            else:
                self.when_left[reviewer] = max(self.when_left[reviewer], review['created_at'])

            if file not in self.when_created:
                self.when_created[file] = review['created_at']
            else:
                self.when_created[file] = min(self.when_created[file], review['created_at'])

        for i, commit in self.data.commits.iterrows():
            file = commit.key_file
            user = commit.key_user

            if user not in self.when_known[file]:
                self.when_known[file][user] = commit['date']
            else:
                prev = self.when_known[file][user]
                self.when_known[file][user] = min(commit['date'], prev)

            if file not in self.when_created:
                self.when_created[file] = commit['date']
            else:
                self.when_created[file] = min(self.when_created[file], commit['date'])

            if user not in self.when_left:
                self.when_left[user] = commit['date']
            else:
                self.when_left[user] = max(self.when_left[user], commit['date'])

    def count(self, history, from_date=None, to_date=None):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        when_known_rev = defaultdict(lambda: {})
        for review in history:
            for file in review['file_path']:
                for reviewer in review['reviewer_login']:
                    if reviewer not in when_known_rev[file]:
                        when_known_rev[file][reviewer] = review['date']
                    else:
                        prev = when_known_rev[file][reviewer]
                        when_known_rev[file][reviewer] = min(review['date'], prev)

        active_dev = {}
        for file in self.when_created:
            if self.when_created[file] > to_date:
                continue
            if file not in active_dev:
                active_dev[file] = set()
            for user in self.when_known[file]:
                if self.when_left[user] >= to_date > self.when_known[file][user]:
                    active_dev[file].add(user)

            for user in when_known_rev[file]:
                if self.when_left[user] >= to_date > when_known_rev[file][user]:
                    active_dev[file].add(user)

        active_dev = {file: len(active_dev[file]) for file in active_dev}
        far = len([file for file in active_dev if active_dev[file] <= 1])

        self.wkr.append(when_known_rev)

        return far, active_dev
