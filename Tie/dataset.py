from Dataset.dataset import DatasetBase


class DS:
    def __init__(self, data):
        self.pulls = data


class TieDataset(DatasetBase):
    def __init__(self, dataset):
        self.start_date = None
        self.end_date = None
        super().__init__(dataset)

    def preprocess(self, dataset):
        pulls = dataset.pulls[dataset.pulls.status != 'OPEN']
        pulls = pulls[['file_path', 'number', 'reviewer_login', 'created_at', 'owner_id', 'comment']].rename(
            {'created_at': 'date', 'comment': 'body'}, axis=1)

        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'date', 'owner_id', 'body']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'date': lambda x: list(x)[0], 'body': lambda x: list(x)[0]}).reset_index()
        pulls = pulls[pulls.reviewer_login.apply(len) > 0]

        pulls = pulls.sort_values('date')

        return pulls

    def replace(self, data, cur_rec):
        pass

    def get_revname(self):
        return 'reviewer_login'

# class TieDataset(DatasetBase):
#     def __init__(self, dataset):
#         super().__init__(dataset)
#         self.pulls = self.data
#
#     def preprocess(self, dataset):
#         pulls = dataset.pulls.rename({'created_at': 'date'}, axis=1)
#
#         pulls = pulls.sort_values('date')
#         return pulls
#
#     def get_revname(self):
#         return 'reviewer_login'
