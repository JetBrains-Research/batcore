from collections import defaultdict

from CounterBase import CounterBase


class CoreWorkloadCounter(CounterBase):
    def count(self, history, from_date=None, to_date=None):
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        review_count = defaultdict(lambda: 0)
        for review in history:
            if review['date'] < from_date:
                continue
            if review['date'] > to_date:
                break

            for reviewer in review['reviewer_login']:
                review_count[review] += 1

        sorted_counts = sorted(review_count.values(), key=lambda x: -x)
        core_workload = sum(sorted_counts[:10])

        return core_workload
