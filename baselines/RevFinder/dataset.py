import numpy as np

from Dataset.StandardDataset import StandardDataset
from Dataset.dataset import DatasetBase
from baselines.Tie.utils import get_all_reviewers


class RevFinderDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.reviewers = get_all_reviewers(data)
