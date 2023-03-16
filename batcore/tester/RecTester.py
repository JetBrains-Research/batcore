import pandas as pd
from tqdm import tqdm

from batcore.tester.TesterBase import TesterBase
from batcore.metrics.metrics import count_metrics
from batcore.bat_logging import tester_logging


class RecTester(TesterBase):
    """
    tester for the standard recommendations metrics
    """

    @tester_logging
    def test_recommender(self,
                         recommender,
                         data_iterator,
                         verbose=False,
                         log_file_path=None,
                         log_stdout=False,
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
        self.info(f"starting evaluation")
        for (train_data, test_data) in tqdm(data_iterator):
            cnt += 1
            recommender.fit(train_data)
            if len(test_data):
                cur_rec = recommender.predict(test_data, n=max(top_ns))
                y_pred = [[*[cur_rec[:n] for n in top_ns], test_data['reviewer'], test_data['key_change']]]
                y_pred = pd.DataFrame(y_pred, columns=[*[f'top-{n}' for n in top_ns], 'rev', 'key'])
                self.recs.append(y_pred)
            self.info(f"finished pull request #{cnt}")

        self.info(f"finished evaluation")
        self.info(f"calculating metrics")
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
                         verbose=False,
                         log_file_path=None,
                         log_stdout=False,
                         *args, **kwargs):
        """
        :param recommender: recommender to be tested. Must implement RecommenderBase interface
        :param data_iterator: iterator over dataset on which recommender will be tested. Must implement
                              IteratorBase interface
        :param top_ns: array of k-s to calculate accuracy@k
        :return: dictionary with acc@k and mean reciprocal rank, recommendations over time
        """
        self.setup_logger(verbose, log_file_path, log_stdout)
        try:
            if top_ns is None:
                top_ns = [1, 3, 5, 10]

            self.recs = []
            cnt = 0
            self.info(f"starting evaluation")
            for (train_data, test_data) in tqdm(data_iterator):
                cnt += 1
                recommender.fit(train_data)
                if len(test_data):
                    cur_rec = recommender.predict(test_data, n=max(top_ns))
                    preds = [[*[cur_rec[:n] for n in top_ns], test_data['reviewer'], test_data[flag],
                              test_data['key_change']]]
                    preds = pd.DataFrame(preds, columns=[*[f'top-{n}' for n in top_ns], 'rev', 'filter_flag', 'key'])
                    self.recs.append(preds)
                self.info(f"finished pull request #{cnt}")
                # if cnt > 50:
                #     # print(np.mean(recommender.log))
                #     break
            self.info(f"finished evaluation")
            self.info(f"calculating metrics")
            recs = pd.concat(self.recs, axis=0)
            recs_filtered = recs[~recs.filter_flag]

            return count_metrics(recs, None, top_ns), count_metrics(recs_filtered, None, top_ns), recs
        except Exception as e:
            self.exception('got an exception')
            raise
