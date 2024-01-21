from copy import deepcopy

from batcore import SimulTester, RecTesterAliasTest
from utils import save_results
from params import *

import pandas as pd

pd.options.mode.chained_assignment = None

models = [ACRec, cHRev, CN, RevFinder,
          RevRec, Tie, WRC, xFinder]


def test_recommendation_metrics(models_cls, path=None, data_path=None, data_args=None, filter_args=None):
    if filter_args is None:
        filter_args = {}
    if data_args is None:
        data_args = {}
    if path is None:
        path = 'results/recmetrics'

    if data_path is None:
        data_path = 'projects/openstack'

    data = GerritLoader(data_path, None, None, True)

    for mdl_cls in models_cls:
        setup = model_setup[mdl_cls.__name__]
        dataset_kwargs = deepcopy(setup['dataset_kwargs'])
        dataset_kwargs.update(data_args)

        dataset = dataset_classes[setup['dataset']](data, **dataset_kwargs)
        print(len(dataset.data))
        data_iterator = iterator_classes[setup['iterator']](dataset, **setup['iterator_kwargs'])

        model_kwargs = deepcopy(setup['model_kwargs'])
        model_kwargs.update(filter_args)

        if setup['item2id']:
            model = mdl_cls(dataset.get_items2ids(), **model_kwargs)
        else:
            model = mdl_cls(**model_kwargs)

        tester = RecTesterAliasTest()
        res = tester.test_recommender(model, data_iterator)
        print(f'Finished for {mdl_cls.__name__}')
        save_results(f'{path}_filtered.json', res[1], model)
        save_results(f'{path}.json', res[0], model)


def test_project_metrics(models_cls, path=None, data_path=None, data_args=None, filter_args=None, dataset_name=''):
    if filter_args is None:
        filter_args = {}
    if data_args is None:
        data_args = {}
    if path is None:
        path = 'results/recmetrics'
    if data_path is None:
        data_path = 'projects/openstack'

    data = GerritLoader(data_path, None, None, True)
    for mdl_cls in models_cls:
        setup = model_setup[mdl_cls.__name__]
        dataset_kwargs = deepcopy(setup['dataset_kwargs'])
        dataset_kwargs.update(data_args)

        dataset = dataset_classes[setup['dataset']](data, **dataset_kwargs)
        data_iterator = PullLoader(dataset, batch_size=1)

        model_kwargs = deepcopy(setup['model_kwargs'])
        model_kwargs.update(filter_args)

        if setup['item2id']:
            model = mdl_cls(dataset.get_items2ids(), **model_kwargs)
        else:
            model = mdl_cls(**model_kwargs)

        # tester = SimulTesterCheckpoint()
        tester = SimulTester()
        res = tester.test_recommender(model, data_iterator, dataset_name=dataset_name)
        print(f'Finished for {mdl_cls.__name__}')
        save_results(path, res, model)


if __name__ == '__main__':
    models = [
        # xFinder,
        # ACRec,
        # cHRev,
        # CN,
        # Tie,
        # RevRec,
        RevFinder,
        # WRC,
    ]
    test_project_metrics(models,
                         path='results/android/proj_alias_base.json',
                         data_path='projects/android/alias',
                         filter_args={'no_owner': True, 'no_inactive': True, 'inactive_time': 30},
                         data_args={'remove_empty': False}, dataset_name='android')

    # test_recommendation_metrics(models,
    #                             path='results/android/rec_base',
    #                             data_path='projects/android/alias',
    #                             filter_args={'no_owner': True, 'no_inactive': True, 'inactive_time': 30},
    #                             data_args={'remove_empty': False})
