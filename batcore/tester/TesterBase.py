from batcore.bat_logging import Logger
from abc import ABC, abstractmethod


class TesterBase(ABC, Logger):
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
                         verbose=False,
                         log_file_path=None,
                         log_stdout=False,
                         log_mode='a',
                         *args, **kwargs):
        self.setup_logger(verbose, log_file_path, log_stdout, log_mode)
