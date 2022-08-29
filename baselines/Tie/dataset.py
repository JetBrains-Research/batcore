import numpy as np

from Dataset.StandardDataset import StandardDataset
from Dataset.dataset import DatasetBase
from baselines.Tie.utils import get_all_reviewers, get_all_words


class DS:
    def __init__(self, data):
        self.pulls = data


class TieDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.reviewers = get_all_reviewers(data)
        self.words = get_all_words(data)
