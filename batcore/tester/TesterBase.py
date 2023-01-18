from abc import ABC, abstractmethod


class TesterBase(ABC):
    """
    Base interface for the tester class

    :param recommender: recommender to be tested
    :param data_iterator: iterator over data on which the recommender will be tested.
    :param args, kwargs: any additional params
    """

    @abstractmethod
    def test_recommender(self,
                         recommender,
                         data_iterator,
                         *args, **kwargs):
        pass
