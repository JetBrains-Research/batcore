import pandas as pd
from tqdm import tqdm

from batcore.tester.TesterBase import TesterBase
from batcore.Metrics.metrics import count_metrics


class RecTester(TesterBase):
    """
    tester for the standard recommendations metrics
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
                y_pred = [[*[cur_rec[:n] for n in top_ns], test_data['reviewer_login'], test_data['key_change']]]
                y_pred = pd.DataFrame(y_pred, columns=[*[f'top-{n}' for n in top_ns], 'rev', 'key'])
                self.recs.append(y_pred)
            # if cnt > 5:
            #     break
        # print(cnt)

        recs = pd.concat(self.recs, axis=0)

        return count_metrics(recs, None, top_ns), recs


class RecTesterAliasTest(TesterBase):
    """
    tester for the standard recommendations metrics
    """

    def test_recommender(self,
                         recommender,
                         data_iterator,
                         flag='self_review',
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
                preds = [[*[cur_rec[:n] for n in top_ns], test_data['reviewer_login'], test_data[flag],
                          test_data['key_change']]]
                preds = pd.DataFrame(preds, columns=[*[f'top-{n}' for n in top_ns], 'rev', 'filter_flag', 'key'])
                self.recs.append(preds)

        recs = pd.concat(self.recs, axis=0)
        recs_filtered = recs[~recs.filter_flag]

        return count_metrics(recs, None, top_ns), count_metrics(recs_filtered, None, top_ns), recs
