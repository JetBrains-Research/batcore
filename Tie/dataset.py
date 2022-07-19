from Dataset.dataset import DatasetBase


class DS:
    def __init__(self, data):
        self.pulls = data


class TieDataset(DatasetBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.pulls = self.data

    def preprocess(self, dataset):
        pulls = dataset.pulls.rename({'created_at': 'date'}, axis=1)

        pulls = pulls.sort_values('date')
        return pulls

    def get_revname(self):
        return 'reviewer_login'

