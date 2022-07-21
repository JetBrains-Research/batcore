from collections import defaultdict

from Counter.CounterBase import CounterBase


class ExpertiseCounter(CounterBase):
    def __init__(self, data):
        self.data = data
        self.prepare()

    def prepare(self):
        self.when_known = defaultdict(lambda: {})

        for i, review in self.data.pulls.iterrows():
            file = review.file_path
            reviewer = review.reviewer_login
            if reviewer not in self.when_known[file]:
                self.when_known[file][reviewer] = review['created_at']
            else:
                prev = self.when_known[file][reviewer]
                self.when_known[file][reviewer] = min(review['created_at'], prev)

        for i, commit in self.data.commits.iterrows():
            file = commit.key_file
            user = commit.key_user
            if user not in self.when_known[file]:
                self.when_known[file][user] = commit['date']
            else:
                prev = self.when_known[file][user]
                self.when_known[file][user] = min(commit['date'], prev)

    def count(self, history, from_date=None, to_date=None):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        expertise = 0
        cnt = 0
        for review in history:
            if review['date'] < from_date:
                continue
            if review['date'] > to_date:
                break

            cnt += 1
            files_known = 0
            if len(review['file_path']) == 0:
                continue

            for file in review['file_path']:
                if file not in self.when_known:
                    continue
                for reviewer in review['reviewer_login']:
                    if reviewer not in self.when_known[file]:
                        continue
                    if self.when_known[file][reviewer] < review['date']:
                        files_known += 1
                        break
            expertise += files_known / len(review['file_path'])

        return expertise
