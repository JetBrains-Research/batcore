from tqdm import tqdm

from Tester.TesterBase import TesterBase


class SimulTester(TesterBase):
    """
    Class for testing non-recommendation metrics on a simulated history
    """

    def __init__(self):
        self.simulated = []
        self.real = []

    def test_recommender(self,
                         recommender,
                         data_iterator,
                         metrics=None,
                         *args, **kwargs):
        """
        :param recommender: recommender to be tested. Must implement RecommenderBase interface
        :param data_iterator: iterator over dataset on which recommender will be tested. Must implement
                              IteratorBase interface
        :param metrics: list of metrics that implement CounterBase interface
        :return: list of calculated metrics
        """

        self.simulate(recommender, data_iterator)

        result = [self.count_metric_dif(metric) for metric in metrics]

        return result

    def simulate(self, recommender, dataset):
        """
        :param recommender: recommender used to simulate reviewer history
        :param dataset: dataset for which history will be simulated
        :return: None. all results are gathered in real and simulated fields
        """
        cnt = 0
        for i, (train_data, test_data) in tqdm(enumerate(dataset)):
            cnt += 1
            if i == 0:
                recommender.fit(train_data)
                self.simulated.append(train_data)
                self.real.append(train_data)

            cur_rec = recommender.predict(test_data, n=1)[0]
            new_train_data = dataset.replace(test_data, cur_rec)

            if i > 0:
                self.simulated.append(new_train_data)
                self.real.append(test_data)
                recommender.fit(new_train_data)

            # if cnt > 1000:
            #     break

    def count_metric_dif(self, metric):
        """
        calculates metric on actual and simulated history and returns its relative difference
        """
        original_metric = metric(self.real)
        simulated_metric = metric(self.simulated)

        dif = (simulated_metric / original_metric - 1) * 100

        return dif
