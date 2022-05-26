from abc import ABC, abstractmethod


class Recommender(ABC):
    def __init__(self):
        pass

    def preprocess(self, data):
        """
        preprocess data for the model if necessary. By default returns data as is
        :param data: data on which the model will be tested
        :return: preprocessed dat
        """
        return data

    @abstractmethod
    def predict(self, data, n=10):
        """
        predicts best n reviewers for each pull in data
        :param data: data for which reviewers need to be predicted
        :param n: number of reviewers to predict
        :return: list of recommendations
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, data):
        """
        trains recommender on given data
        :param data: train data to fit
        """
        raise NotImplementedError
