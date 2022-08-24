import numpy as np

from Dataset.dataset import DatasetBase


class DS:
    def __init__(self, data):
        self.pulls = data


class TieDataset(DatasetBase):
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
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'owner', 'comment']].rename(
            {'created_at': 'date', 'comment': 'body'}, axis=1)

        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'owner', 'body']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0], 'body': lambda x: list(x)[0], 'owner': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]
        pulls['type'] = 'pull'
        pulls = pulls.sort_values('date')
        pulls = pulls.to_dict('records')
        return pulls

    def replace(self, data, cur_rec):
        pass

    def get_revname(self):
        return 'reviewer_login'
