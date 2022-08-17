from datetime import datetime

import numpy as np

from Dataset.dataset import DatasetBase
from baselines.WCS.utils import ItemMap


class CNDataset(DatasetBase):
    def __init__(self, dataset, max_file=np.inf):
        """
        :param max_file: maximum number of files that a review can have
        """
        self.max_file = max_file
        super().__init__(dataset)

    def preprocess(self, dataset, to_date=datetime(year=2017, month=1, day=1)):
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        pulls = pulls[pulls.created_at < to_date]
        pulls = pulls[['file_path', 'key_change', 'reviewer_login', 'created_at', 'owner']].rename(
            {'created_at': 'date'}, axis=1)

        pulls = pulls.groupby('key_change')[['file_path', 'reviewer_login', 'date', 'owner']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0], 'owner': lambda x: list(x)[0]}).reset_index()

        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]
        pulls['type'] = 'pull'

        comments = dataset.comments[dataset.comments.date < to_date]
        comments['type'] = 'comment'

        data = pulls.to_dict('records') + comments.to_dict('records')
        data = sorted(data, key=lambda x: x['date'])

        self.users = ItemMap(pulls['reviewer_login'].sum() + pulls['owner'].to_list() + comments['key_user'].to_list())

        return data

    def replace(self, data, cur_rec):
        pass

    def get_revname(self):
        return 'reviewer_login'

