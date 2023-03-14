import logging
import sys
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
                         verbose=False,
                         log_file_path=None,
                         log_stdout=False,
                         *args, **kwargs):
        self.setup_logger(verbose, log_file_path, log_stdout)

    def setup_logger(self, verbose, log_file_path, log_stdout):
        self.verbose = verbose
        if self.verbose:
            if (log_file_path is None) and (not log_stdout):
                raise ValueError("When verbose=True you need to specify log_file_path or set log_stdout to True")

            handlers = []
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            if log_file_path is not None:
                file_handler = logging.FileHandler(log_file_path, f)
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            if log_stdout:
                stdout_handler = logging.StreamHandler(sys.stdout)
                stdout_handler.setFormatter(formatter)
                handlers.append(stdout_handler)

            self.logger = logging.getLogger(type(self).__name__)
            for handler in handlers:
                self.logger.addHandler(handler)

            self.logger.setLevel(logging.INFO)

    def info(self, msg):
        if self.verbose:
            self.logger.info(msg)

    def exception(self, msg):
        if self.verbose:
            self.logger.exception(msg)

    def warning(self, msg):
        if self.verbose:
            self.logger.warning(msg)

    def error(self, msg):
        if self.verbose:
            self.logger.error(msg)

    def debug(self, msg):
        if self.verbose:
            self.logger.debug(msg)
