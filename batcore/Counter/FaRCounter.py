from collections import defaultdict
from itertools import chain

from batcore.Counter.CounterBase import CounterBase


class FaRCounter(CounterBase):
    """
    Files as risk metric = number of files that are known by one or zero active developers
    """

    def __init__(self, iterator):
        self.when_known = defaultdict(lambda: {})
        self.when_left = {}
        self.when_created = {}

        self.data = iterator.data
        self.prepare()

        self.wkr = []

    def prepare(self):
        """
        supporting calculation for the faster metric estimation
        """

        for event in self.data:
            if event['type'] == 'pull':
                for user in chain(event['reviewer_login'], event['owner']):
                    if user not in self.when_left:
                        self.when_left[user] = event['date']
                    else:
                        self.when_left[user] = max(self.when_left[user], event['date'])

                for file in event['file_path']:
                    if file not in self.when_created:
                        self.when_created[file] = event['date']
                    else:
                        self.when_created[file] = min(self.when_created[file], event['date'])


            elif event['type'] == 'commit' or event['type'] == 'comment':
                file = event['key_file']
                user = event['key_user']

                if user not in self.when_known[file]:
                    self.when_known[file][user] = event['date']
                else:
                    prev = self.when_known[file][user]
                    self.when_known[file][user] = min(event['date'], prev)

                if file not in self.when_created:
                    self.when_created[file] = event['date']
                else:
                    self.when_created[file] = min(self.when_created[file], event['date'])

                if user not in self.when_left:
                    self.when_left[user] = event['date']
                else:
                    self.when_left[user] = max(self.when_left[user], event['date'])

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

        when_known_rev = defaultdict(lambda: {})
        for pull in history:
            for file in pull['file_path']:
                for reviewer in pull['reviewer_login']:
                    if reviewer not in when_known_rev[file]:
                        when_known_rev[file][reviewer] = pull['date']
                    else:
                        prev = when_known_rev[file][reviewer]
                        when_known_rev[file][reviewer] = min(pull['date'], prev)

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
                try:
                    if self.when_left[user] >= to_date > when_known_rev[file][user]:
                        active_dev[file].add(user)
                except Exception as e:
                    print(file, user)
                    raise e

        active_dev = {file: len(active_dev[file]) for file in active_dev}
        far = len([file for file in active_dev if active_dev[file] <= 1])

        self.wkr.append(when_known_rev)

        return far  # , active_dev
