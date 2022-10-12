from datetime import datetime

import numpy as np
import pandas as pd

from Dataset.StandardDataset import StandardDataset
from Dataset.dataset import DatasetBase
from baselines.RevRec.utils import ItemMap


class RevRecDataset(StandardDataset):
    def additional_preprocessing(self, events, data):
        self.users = ItemMap()
        for event in data:
            if event['type'] == 'pull':
                for rev in event['reviewer_login']:
                    self.users.add2(rev)
        for user in pd.unique(events['pulls']['owner'].sum()):
            self.users.add2(user)

        for user in pd.unique(events['comments']['key_user']):
            self.users.add2(user)
