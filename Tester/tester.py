import pandas as pd
from tqdm import tqdm

from Tester.TesterBase import TesterBase
from utils import count_metrics


class Tester(TesterBase):
    """
    Tester for the standard recommendations metrics
    """
    def test_recommender(self,
                         recommender,
                         data_iterator,
                         top_ns=None,
                         *args, **kwargs):
        """
        :param recommender: recommender to be tested. Must implement RecommenderBase interface
        :param data_iterator: iterator over dataset on which recommender will be tested. Must implement
                              IteratorBase interface
        :param top_ns: array of k-s to calculate accuracy@k
        :return: dictionary with acc@k and mean reciprocal rank, recommendations over time
        """
        if top_ns is None:
            top_ns = [1, 3, 5, 10]

        self.recs = []
        cnt = 0
        for (train_data, test_data) in tqdm(data_iterator):
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
