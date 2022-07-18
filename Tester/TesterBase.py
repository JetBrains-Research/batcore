from abc import ABC, abstractmethod

import pandas as pd


class TesterBase(ABC):
    @abstractmethod
    def test_recommender(self,
                         recommender,
                         dataset,
                         *args, **kwargs):
        pass