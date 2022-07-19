from collections import defaultdict

from Counter.CounterBase import CounterBase


class FaRCounter(CounterBase):
    @classmethod
    def count(cls, history, from_date=None, to_date=None):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        when_created = defaultdict(lambda: None)
        when_left = defaultdict(lambda: None)
        is_known = defaultdict(lambda: set())

        for review in history:
            for file in review['file_path']:
                if file not in when_created:
                    when_created[file] = review['date']
                else:
                    when_created[file] = min(when_created[file], review['date'])

                for reviewer in review['reviewer_login']:
                    is_known[file].add(reviewer)

            for reviewer in review['reviewer_login']:
                if reviewer not in when_left:
                    when_left[reviewer] = review['date']
                else:
                    when_left[reviewer] = max(when_left[reviewer], review['date'])

        active_dev = {}
        for review in history:
            if review['date'] > to_date:
                break

            for file in review['file_path']:
                if file not in active_dev:
                    active_dev[file] = set()
                for reviewer in is_known[file]:
                    if when_left[reviewer] >= to_date:
                        active_dev[file].add(reviewer)
        active_dev = {file: len(active_dev[file]) for file in active_dev}
        far = len([file for file in active_dev if active_dev[file] <= 1])

        return far



