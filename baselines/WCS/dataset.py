import numpy as np

from Dataset.dataset import DatasetBase
from .utils import ItemMap


def remove_owner(row):
    owner = row.owner
    revs = row.reviewer_login
    if owner in revs:
        revs.remove(owner)
    return revs


class WCSDataset(DatasetBase):
    def __init__(self, dataset, max_file=np.inf):
        """
        :param max_file: maximum number of files that a review can have
        """
        self.start_date = None
        self.end_date = None
        self.max_file = max_file

        super().__init__(dataset)

    def preprocess(self, dataset):
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'owner']].rename(
            {'created_at': 'date'}, axis=1)

        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'owner']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0], 'owner': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]
        pulls['type'] = 'pull'
        pulls['reviewer_login'] = pulls.apply(remove_owner, axis=1)
        self.pulls = pulls

        self.users = ItemMap(pulls['reviewer_login'].sum())
        self.files = ItemMap(pulls['file_path'].sum())

        data = pulls.to_dict('records')
        return data

    def replace(self, data, cur_rec):
        pass

    def get_revname(self):
        return 'reviewer_login'
