from copy import deepcopy

from tqdm import tqdm

from Counter.CoreWorkloadCounter import CoreWorkloadCounter
from Counter.ExpertiseCounter import ExpertiseCounter
from Counter.FaRCounter import FaRCounter
from tester.TesterBase import TesterBase


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
        :param metrics: dict of metrics that implement CounterBase interface
        :return: list of calculated metrics
        """

        if metrics is None:
            metrics = {'Core Workload': CoreWorkloadCounter,
                       'FaR': FaRCounter(data_iterator),
                       'Expertise': ExpertiseCounter(data_iterator)}
        self.simulate(recommender, data_iterator)

        result = {metric_name: self.count_metric_dif(metrics[metric_name]) for metric_name in metrics}

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
                for event in train_data:
                    if event['type'] == 'pull':
                        self.simulated.append(event)
                        self.real.append(event)

            recommender.fit(train_data)

            cur_rec = recommender.predict(test_data, n=5)
            self.real.append(deepcopy(test_data))

            if len(cur_rec):
                simulated_pull = dataset.replace(cur_rec[0])
                self.simulated.append(simulated_pull)
            else:
                self.simulated.append(test_data)

            if cnt > 200:
                break

    def count_metric_dif(self, metric):
        """
        calculates metric on actual and simulated history and returns its relative difference
        """
        original_metric = metric(self.real)
        simulated_metric = metric(self.simulated)
        # print(original_metric, simulated_metric)
        dif = (simulated_metric / original_metric - 1) * 100

        return dif
