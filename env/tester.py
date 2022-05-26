from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

from recommender import *
from recsys.utils import count_metrics


class TesterBase(ABC):
    def __init__(self, data):
        self.raw_data = data

    @abstractmethod
    def test_recommender(self, recommender,
                         initial_delta, test_interval):
        pass


class Tester(TesterBase):
    def __init__(self, data):
        super().__init__(data)

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

        test_interval = timedelta(7, 0)
        recs = []
        for i in range(np.ceil((end_date - to_date) / test_interval)):
            test_date = to_date + test_interval
            test_data = [df[(df.date >= to_date) & (df.date < test_date)] for df in data]

            recommender.fit(train_data)
            recs.append(recommender.predict(test_data, n=max(top_ns)))

            train_data = test_data
            to_date = test_date
            if to_date > end_date:
                break

        recs = pd.concat(recs, axis=0)

        return count_metrics(recs)
