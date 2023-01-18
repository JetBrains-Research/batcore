from abc import ABC, abstractmethod


class CounterBase(ABC):
    """
    Base class for non-recommendation metrics with a history simulation

    :param history: historical data
    """
    @abstractmethod
    def __call__(self, history, *args, **kwargs):

        pass
