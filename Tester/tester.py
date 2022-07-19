import pandas as pd
from tqdm import tqdm

from utils import count_metrics


class Tester:

    def test_recommender(self,
                         recommender,
                         dataset,
                         top_ns=None):
        if top_ns is None:
            top_ns = [1, 3, 5, 10]

        self.recs = []
        cnt = 0
        for (train_data, test_data) in tqdm(dataset):
            cnt += 1
            recommender.fit(train_data)
            if len(test_data):
                cur_rec = recommender.predict(test_data, n=max(top_ns))
                y_pred = [[*[cur_rec[:n] for n in top_ns], test_data['reviewer_login']]]
                y_pred = pd.DataFrame(y_pred, columns=[*[f'top-{n}' for n in top_ns], 'rev'])
                self.recs.append(y_pred)
            if cnt > 200:
                break
        # print(cnt)

        recs = pd.concat(self.recs, axis=0)

        return count_metrics(recs), recs
