from collections import defaultdict

from nltk import PorterStemmer, RegexpTokenizer

from Dataset.dataset import DatasetBase
from Tie.utils import tokenize, vectorize


class TieDataset(DatasetBase):
    def __init__(self, dataset, initial_delta=None, test_interval=None):
        self.start_date = None
        self.end_date = None
        super().__init__(dataset, initial_delta, test_interval)

        self.pulls = self.data

    def preprocess(self, dataset):
        pulls = dataset.pulls[['file_path', 'created_at', 'number', 'author_login', 'reviewer_login', 'body']]
        # pulls = dataset.pulls[['file_path', 'created_at', 'number', 'reviewer_login', 'body']]
        ps = PorterStemmer()

        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'created_at', 'author_login', 'body']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)), 'created_at': lambda x: list(x)[0],
             'body': lambda x: x.iloc[0]}).reset_index()

        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r"\s+|/|\\|\.s|.$", gaps=True)

        pulls['body'] = pulls.body.apply(lambda x: tokenize(x, tokenizer, stemmer))
        df = defaultdict(lambda: 0)

        for b in pulls['body']:
            for t in b:
                df[t] += 1
        tok2id = {}

        cur_id = 0
        for b in pulls['body']:
            for t in b:
                if t in tok2id or df[t] < 2:
                    continue
                else:
                    tok2id[t] = cur_id
                    cur_id += 1
        pulls['body'] = pulls.body.apply(lambda x: vectorize(x, tok2id))
        pulls = pulls.rename({'created_at': 'date'}, axis=1)
        self.start_date = pulls.date.min()
        self.end_date = pulls.date.max()
        return pulls

    def __iter__(self):
        self.to_date = self.initial_delta + self.start_date
        self.test_date = self.to_date + self.test_interval

        self.train = self.pulls[(self.pulls.date < self.to_date) & (self.pulls.date >= self.start_date)]
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        return self

    def __next__(self):
        train = self.train
        test = self.test

        self.train = self.test
        self.test = self.pulls[(self.pulls.date >= self.to_date) & (self.pulls.date < self.test_date)]

        self.to_date = self.test_date
        self.test_date = self.to_date + self.test_interval
        if self.to_date > self.end_date:
            raise StopIteration

        return train, test
