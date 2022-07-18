from collections import defaultdict

from nltk import PorterStemmer, RegexpTokenizer

from Dataset.dataset import DatasetBase
from Tie.utils import tokenize, vectorize


class TieDataset(DatasetBase):
    def __init__(self, dataset):

        super().__init__(dataset)

    def preprocess(self, dataset):
        pulls = dataset.pulls[['file_path', 'created_at', 'number', 'author_login', 'reviewer_login', 'body']]
        # pulls = dataset.pulls[['file_path', 'created_at', 'number', 'reviewer_login', 'body']]

        pulls = pulls.groupby('number')[['file_path', 'reviewer_login', 'created_at', 'author_login', 'body']].agg(
            {'file_path': lambda x: list(set(x)), 'reviewer_login': lambda x: list(set(x)),
             'created_at': lambda x: list(x)[0],
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
        return pulls
