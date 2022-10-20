import numpy as np

from Dataset.StandardDataset import StandardDataset
from baselines.Tie.utils import get_all_reviewers


class RevFinderDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.reviewers = get_all_reviewers(data)

    def get_items2ids(self):
        ret = {'reviewers': self.reviewers}
        return ret
