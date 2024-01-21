import os
import pickle
from copy import deepcopy

from tqdm import tqdm

from batcore.counter import CoreWorkloadCounter
from batcore.counter import FaRCounter, ExpertiseCounter
from batcore.tester.TesterBase import TesterBase

from batcore.bat_logging import tester_logging


class SimulTester(TesterBase):
    """
    Class for testing non-recommendation metrics on a simulated history
    """

    def __init__(self):
        self.simulated = []
        self.real = []

    @tester_logging
    def test_recommender(self,
                         recommender,
                         data_iterator,
                         metrics=None,
                         dataset_name='',
                         from_checkpoint=False,
                         *args, **kwargs):
        """
        :param recommender: recommender to be tested. Must implement RecommenderBase interface
        :param data_iterator: iterator over dataset on which recommender will be tested. Must implement
                              IteratorBase interface
        :param metrics: dict of metrics that implement CounterBase interface
        :return: list of calculated metrics
        """

        self.info(f"starting evaluation")
        if metrics is None:
            metrics = {'Core Workload': CoreWorkloadCounter(),
                       'FaR': FaRCounter(data_iterator),
                       'Expertise': ExpertiseCounter(data_iterator)}
        if not from_checkpoint:
            self.simulate(recommender, data_iterator)
            self.info(f"saving simulation to checkpoints/{dataset_name}/{type(recommender).__name__}")
            self.save(path=f'checkpoints/{dataset_name}/{type(recommender).__name__}')
        else:
            self.info(f"loading simulation from checkpoints/{dataset_name}/{type(recommender).__name__}")
            self.load(path=f'checkpoints/{dataset_name}/{type(recommender).__name__}')

        self.info(f"finished evaluation")
        self.info(f"calculating metrics")

        result = {metric_name: self.count_metric_dif(metrics[metric_name]) for metric_name in metrics}

        return result

    def simulate(self, recommender, dataset):
        """
        :param recommender: recommender used to simulate reviewer history
        :param dataset: dataset for which history will be simulated
        :return: None. all results are gathered in real and simulated fields
        """
        cnt = 0
        self.info(f"starting simulation")
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

            self.info(f"finished simulation for pull request #{cnt}")

    def count_metric_dif(self, metric):
        """
        calculates metric on actual and simulated history and returns its relative difference
        """
        original_metric = metric(self.real)
        simulated_metric = metric(self.simulated)
        # print(original_metric, simulated_metric)
        dif = (simulated_metric / original_metric - 1) * 100

        return dif

    def save(self, path='checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}/simulated.pkl', 'wb') as f:
            pickle.dump(self.simulated, f)

        with open(f'{path}/real.pkl', 'wb') as f:
            pickle.dump(self.real, f)

    def load(self, path='checkpoints'):
        with open(f'{path}/simulated.pkl', 'rb') as f:
            self.simulated = pickle.load(f)

        with open(f'{path}/real.pkl', 'rb') as f:
            self.real = pickle.load(f)


class SimulTesterSingleSkip(SimulTester):
    """
    Class for testing non-recommendation metrics on a simulated history
    """

    def simulate(self, recommender, dataset):
        """
        :param recommender: recommender used to simulate reviewer history
        :param dataset: dataset for which history will be simulated
        :return: None. all results are gathered in real and simulated fields
        """
        cnt = 0
        self.info(f"starting simulation")
        for i, (train_data, test_data) in tqdm(enumerate(dataset)):
            cnt += 1
            if i == 0:
                for event in train_data:
                    if event['type'] == 'pull':
                        self.simulated.append(event)
                        self.real.append(event)

            recommender.fit(train_data)

            self.real.append(deepcopy(test_data))

            if len(test_data['reviewer']) > 1:
                cur_rec = recommender.predict(test_data, n=5)
                if len(cur_rec) > 0:
                    simulated_pull = dataset.replace(cur_rec[0])
                    self.simulated.append(simulated_pull)
                else:
                    self.simulated.append(test_data)
            else:
                self.simulated.append(test_data)
            self.info(f"finished simulation for pull request #{cnt}")

    def count_metric_dif(self, metric):
        """
        calculates metric on actual and simulated history and returns its relative difference
        """
        original_metric = metric(self.real)
        simulated_metric = metric(self.simulated)
        # print(original_metric, simulated_metric)
        dif = (simulated_metric / original_metric - 1) * 100

        return dif

    def save(self, path='checkpoints'):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}/simulated.pkl', 'wb') as f:
            pickle.dump(self.simulated, f)

        with open(f'{path}/real.pkl', 'wb') as f:
            pickle.dump(self.real, f)

    def load(self, path='checkpoints'):
        with open(f'{path}/simulated.pkl', 'rb') as f:
            self.simulated = pickle.load(f)

        with open(f'{path}/real.pkl', 'rb') as f:
            self.real = pickle.load(f)


class SimulTesterCheckpoint(TesterBase):
    """
    Class for testing non-recommendation metrics on a simulated history
    """

    def __init__(self):
        self.simulated = []
        self.real = []

    @tester_logging
    def test_recommender(self,
                         recommender,
                         data_iterator,
                         metrics=None,
                         from_checkpoint=True,
                         *args, **kwargs):
        """
        :param recommender: recommender to be tested. Must implement RecommenderBase interface
        :param data_iterator: iterator over dataset on which recommender will be tested. Must implement
                              IteratorBase interface
        :param metrics: dict of metrics that implement CounterBase interface
        :return: list of calculated metrics
        """

        if metrics is None:
            metrics = {'Core Workload': CoreWorkloadCounter(),
                       'FaR': FaRCounter(data_iterator),
                       'Expertise': ExpertiseCounter(data_iterator)}
        self.simulate(recommender, data_iterator, from_checkpoint)

        result = {metric_name: self.count_metric_dif(metrics[metric_name]) for metric_name in metrics}

        return result

    def simulate(self, recommender, dataset, from_checkpoint):
        """
        :param recommender: recommender used to simulate reviewer history
        :param dataset: dataset for which history will be simulated
        :return: None. all results are gathered in real and simulated fields
        """
        cnt = 0
        if from_checkpoint:
            self.load(recommender)
        old_cnt = len(self.real)
        for i, (train_data, test_data) in tqdm(enumerate(dataset)):
            cnt += 1
            if cnt < old_cnt:
                continue
            if cnt % 500 == 0:
                self.save(recommender)
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

            if cnt > 100:
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

    def save(self, model, path='checkpoints/libre'):
        model.save()
        with open(f'{path}/simulated.pkl', 'wb') as f:
            pickle.dump(self.simulated, f)

        with open(f'{path}/real.pkl', 'wb') as f:
            pickle.dump(self.real, f)

    def load(self, model, path='checkpoints/libre'):
        model.load()
        with open(f'{path}/simulated.pkl', 'rb') as f:
            self.simulated = pickle.load(f)

        with open(f'{path}/real.pkl', 'rb') as f:
            self.real = pickle.load(f)
