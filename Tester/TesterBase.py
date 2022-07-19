from abc import ABC, abstractmethod


class TesterBase(ABC):
    @abstractmethod
    def test_recommender(self,
                         recommender,
                         dataset,
                         *args, **kwargs):
        pass
