from copy import deepcopy

from Counter.CoreWorkloadCounter import CoreWorkloadCounter
from Counter.ExpertiseCounter import ExpertiseCounter
from Counter.FaRCounter import FaRCounter
from baselines import *
from data import *
from tester import RecTester, SimulTester
from utils import save_results
from params import *

import pandas as pd

pd.options.mode.chained_assignment = None

models = [ACRec, cHRev, CN, RevFinder,
          RevRec, Tie, WRC, xFinder]


def test_recommendation_metrics(models_cls, path=None, data_args=None, filter_args=None):
    if filter_args is None:
        filter_args = {}
    if data_args is None:
        data_args = {}
    if path is None:
        path = 'results/recmetrics'

    data = GerritLoader('projects/openstack', None, None, True)

    for mdl_cls in models_cls:
        setup = model_setup[mdl_cls.__name__]
        dataset_kwargs = setup['dataset_kwargs'].update(data_args)

        dataset = dataset_classes[setup['data']](data, **dataset_kwargs)
        data_iterator = iterator_classes[setup['iterator']](**setup['iterator_args'])

        model_kwargs = setup['model_kwargs'].update(filter_args)
        if setup['items2id']:
            model = mdl_cls(dataset.get_items2ids, **model_kwargs)
        else:
            model = mdl_cls(**model_kwargs)

        tester = RecTester()
        res = tester.test_recommender(model, data_iterator)
        print(f'Finished for {mdl_cls.__name__}')
        save_results(path, res[0], model)


def test_project_metrics(models_cls, path=None, data_args=None, filter_args=None):
    if filter_args is None:
        filter_args = {}
    if data_args is None:
        data_args = {}
    if path is None:
        path = 'results/recmetrics'

    data = GerritLoader('projects/openstack', None, None, True)

    for mdl_cls in models_cls:
        setup = model_setup[mdl_cls.__name__]
        dataset_kwargs = deepcopy(setup['dataset_kwargs'])
        dataset_kwargs.update(data_args)

        dataset = dataset_classes[setup['data']](data, **dataset_kwargs)
        data_iterator = StreamUntilLoader(dataset)

        model_kwargs = deepcopy(setup['model_kwargs'])
        model_kwargs.update(filter_args)

        if setup['item2id']:
            model = mdl_cls(dataset.get_items2ids(), **model_kwargs)
        else:
            model = mdl_cls(**model_kwargs)

        tester = SimulTester()
        res = tester.test_recommender(model, data_iterator)
        print(f'Finished for {mdl_cls.__name__}')
        save_results(path, res, model)


if __name__ == '__main__':
    models = [
        # ACRec,
        # cHRev,
        # CN,
        # xFinder,
        # Tie,

        # RevRec,
        # RevFinder,
        WRC,
    ]
    test_project_metrics(models, 'results/openstack_project_metrics.json', data_args={'commits': True})
