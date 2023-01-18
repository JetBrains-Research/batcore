from abc import ABC, abstractmethod


class DatasetBase(ABC):
    """
    Base class for a dataset that encapsulates data preprocessing
    """

    def __init__(self, data):
        """
        :param data: Data to be stored in a dataset
        """
        self.data = self.preprocess(data)

    @abstractmethod
    def preprocess(self, data):
        """
        performs any necessary preprocessing needed
        """
        pass

    def replace(self, data, cur_rec):
        """
        A method that is used for simulating history
        :param data: a single pull that needed to ba modified
        :param cur_rec: reviewer to be added into a pull
        :return: pull with a modified reviewer list
        """
        raise NotImplementedError()


