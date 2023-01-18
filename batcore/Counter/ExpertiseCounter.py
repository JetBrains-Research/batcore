from collections import defaultdict

from batcore.Counter.CounterBase import CounterBase


class ExpertiseCounter(CounterBase):
    """
    Expertise metric is estimated as an average proportion of the files under that the selected reviewers modified of
    reviewed in the past
    """

    def __init__(self, iterator):
        self.when_known = None
        self.data = iterator.data
        self.prepare()

    def prepare(self):
        """
        supporting calculation for the faster metric estimation
        """
        self.when_known = defaultdict(lambda: {})

        for event in self.data:
            if event['type'] == 'pull':
                for file in event['file_path']:
                    for reviewer in event['reviewer_login']:
                        if reviewer not in self.when_known[file]:
                            self.when_known[file][reviewer] = event['date']
                        else:
                            prev = self.when_known[file][reviewer]
                            self.when_known[file][reviewer] = min(event['date'], prev)
            elif event['type'] == 'commit' or event['type'] == 'comment':
                file = event['key_file']
                user = event['key_user']
                if user not in self.when_known[file]:
                    self.when_known[file][user] = event['date']
                else:
                    prev = self.when_known[file][user]
                    self.when_known[file][user] = min(event['date'], prev)

    def __call__(self, history, from_date=None, to_date=None):
        """
           :param history: data with reviews
           :param from_date: start of the period on which CoreWorkload is calculated.
                             If None starts from the start of history
           :param to_date: end of the period on which CoreWorkload is calculated.
                           If None ends at the end of history
           :return: Expertise metrics
        """
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        expertise = 0
        cnt = 0
        for pull in history:
            if pull['date'] < from_date:
                continue
            if pull['date'] > to_date:
                break

            cnt += 1
            files_known = 0
            if len(pull['file_path']) == 0:
                continue

            for file in pull['file_path']:
                if file not in self.when_known:
                    continue
                for reviewer in pull['reviewer_login']:
                    if reviewer not in self.when_known[file]:
                        continue
                    if self.when_known[file][reviewer] < pull['date']:
                        files_known += 1
                        break
            expertise += files_known / len(pull['file_path'])

        return expertise
