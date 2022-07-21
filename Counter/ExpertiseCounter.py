from collections import defaultdict

from Counter.CounterBase import CounterBase


class ExpertiseCounter(CounterBase):
    """
    Expertise metric. Calculates percentage of files in a review known to the reviewers
    """
    @classmethod
    def count(cls, history, from_date=None, to_date=None):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        when_known = defaultdict(lambda: {})
        for review in history:
            for file in review['file_path']:
                for reviewer in review['reviewer_login']:
                    if reviewer not in when_known[file]:
                        when_known[file][reviewer] = review['date']
                    else:
                        prev = when_known[file][reviewer]
                        when_known[file][reviewer] = min(review['date'], prev)

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
                if file not in when_known:
                    continue
                for reviewer in review['reviewer_login']:
                    if reviewer not in when_known[file]:
                        continue
                    if when_known[file][reviewer] < review['date']:
                        files_known += 1
                        break
            expertise += files_known / len(review['file_path'])

        return expertise


