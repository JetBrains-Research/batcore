from abc import ABC, abstractmethod


class RecommenderBase(ABC):
    def __init__(self):
        pass

    def preprocess(self, data):
        """
        preprocess Dataset for the model if necessary. Should return any iterable object of dataframes used during
        training. By default returns Dataset as is
        :param data: Dataset on which the model will be tested
        :return: preprocessed Dataset
        """
        return data

    @abstractmethod
    def predict(self, data, n=10):
        """
        predicts best n reviewers for each pull in Dataset
        :param data: Dataset for which reviewers need to be predicted
        :param n: number of reviewers to predict
        :return: list of recommendations
        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        trains RecommenderBase on given Dataset
        :param data: train Dataset to fit
        """
        pass