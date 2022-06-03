from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

from utils import count_metrics
from .recommender import *


class TesterBase(ABC):
    def __init__(self, data,
                 prepare: callable = None):
        if prepare is None:
            self.raw_data = self.prepare(data)
        else:
            self.raw_data = prepare(data)

    @abstractmethod
    def test_recommender(self, recommender,
                         initial_delta, test_interval):
        pass

    @abstractmethod
    def prepare(self, data):
        pass


class Tester(TesterBase):
    def __init__(self, data):
        super().__init__(data)

    def prepare(self, data):
        pulls = data['pull_file'].merge(data['reviewer']).merge(data['pull'][['number', 'created_at']],
                                                                left_on='pull_number',
                                                                right_on='number').merge(data['pull_author']).dropna()
        pulls['date'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)
        commits = data['commit_file'].merge(data['commit_author']).merge(data['commit'][['oid', 'committed_date']],
                                                                         left_on='oid', right_on='oid').dropna()
        commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)

        pulls = pulls.drop(['pull_number', 'created_at'], axis=1)
        commits = commits.drop(['oid', 'file_oid', 'committed_date'], axis=1)

        return pulls, commits

    # TODO add col for date as parameter
    def test_recommender(self,
                         recommender: RecommenderBase,
                         initial_delta: Union[int, str] = 'auto',
                         test_interval: int = 7,
                         top_ns: list = None):
        if top_ns is None:
            top_ns = [1, 3, 5, 10]

        # prepare data

        data = recommender.preprocess(self.raw_data)

        # initial dates
        from_date = None
        end_date = None

        for df in data:
            from_date = df.date.min() if from_date is None else min(from_date, df.date.min())
            end_date = df.date.max() if end_date is None else max(end_date, df.date.max())

        if isinstance(initial_delta, int):
            to_date = timedelta(initial_delta, 0) + from_date
        elif isinstance(initial_delta, str):
            # TODO
            raise NotImplementedError
        else:
            raise ValueError(f"Wrong initial_delta. Should be an int or 'auto'")

        train_data = [df[(df.date < to_date) & (df.date >= from_date)] for df in data]

        test_interval = timedelta(test_interval, 0)

        recs = []
        for _ in range(int(np.ceil((end_date - to_date) / test_interval))):
            test_date = to_date + test_interval
            test_data = [df[(df.date >= to_date) & (df.date < test_date)] for df in data]
            pred_data = test_data[0]
            pred_data = pred_data.groupby('number')[['file_path', 'reviewer_login']].agg(
                {'file_path': list, 'reviewer_login': lambda x: list(set(x))}).reset_index()

            recommender.fit(train_data)
            cur_rec = recommender.predict(pred_data, n=max(top_ns))
            y_pred = []
            for i, row in pred_data.iterrows():
                y_pred.append([row.number, *[cur_rec[i][:n] for n in top_ns], row.reviewer_login])
            y_pred = pd.DataFrame(y_pred, columns=['number', *[f'top-{n}' for n in top_ns], 'rev'])
            recs.append(y_pred)

            train_data = test_data
            to_date = test_date
            if to_date > end_date:
                break

        recs = pd.concat(recs, axis=0)

        return count_metrics(recs)
