from abc import ABC, abstractmethod


class CounterBase(ABC):
    """
    Base class for non-recommendation metrics with a history simulation
    """
    @abstractmethod
    def count(self, history, *args, **kwargs):
        """
        :param history: historical data
        """
        pass
