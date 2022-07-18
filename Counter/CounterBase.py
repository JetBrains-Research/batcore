from abc import ABC, abstractmethod


class CounterBase(ABC):
    @abstractmethod
    def count(self, history, *args, **kwargs):
        pass
