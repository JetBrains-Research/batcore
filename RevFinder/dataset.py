import numpy as np

from Dataset.dataset import DatasetBase


class RevFinderDataset(DatasetBase):
    def __init__(self, dataset, max_file=np.inf):
        """
        :param max_file: maximum number of files that a review can have
        """
        self.start_date = None
        self.end_date = None
        self.max_file = max_file
        super(RevFinderDataset, self).__init__(dataset)
        # self.log = defaultdict(lambda: [])

    def preprocess(self, dataset):
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'owner_id']].rename(
            {'created_at': 'date'}, axis=1)
        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'owner_id']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]
        pulls = pulls[pulls.file_path.apply(len) <= self.max_file]

        # self.id2file = list(dict.fromkeys(pulls.file_path.sum()))
        # self.file2id = {f: i for i, f in enumerate(self.id2file)}

        return pulls

    def replace(self, data, cur_rec):
        pass

    def get_revname(self):
        return 'reviewer_login'
