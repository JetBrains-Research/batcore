from collections import defaultdict

from batcore.counter.CounterBase import CounterBase


class CoreWorkloadCounter(CounterBase):

    def __call__(self, history, from_date=None, to_date=None):

        """
        Core Workload estimates amount of work for core developers. Core Workload equals to the total number of reviews
        performed by 10 reviews that reviewed the most

        :param history: data with reviews
        :param from_date: start of the period on which CoreWorkload is calculated. If None starts from the start of history
        :param to_date: end of the period on which CoreWorkload is calculated. If None ends at the end of history
        :return: Core Workload metrics
        """
        if from_date is None:
            from_date = history[0]['date']
        if to_date is None:
            to_date = history[-1]['date']

        review_count = defaultdict(lambda: 0)
        for pull in history:
            if pull['date'] < from_date:
                continue
            if pull['date'] > to_date:
                break

            for reviewer in pull['reviewer']:
                review_count[reviewer] += 1

        sorted_counts = sorted(review_count.values(), key=lambda x: -x)
        core_workload = sum(sorted_counts[:10])

        return core_workload
