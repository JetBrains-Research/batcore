from abc import ABC, abstractmethod


class CounterBase(ABC):
    """
    Base class for metrics on a simulated history
    """
    @abstractmethod
    def count(self, history, *args, **kwargs):
        pass
