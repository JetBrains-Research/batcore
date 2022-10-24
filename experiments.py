from baselines import *
from dataset import *
from tester import RecTester
from utils import save_results
from params import *

import pandas as pd

pd.options.mode.chained_assignment = None

data = GerritLoader('projects/openstack', None, None, True)

models = [ACRec, cHRev, CN, RevFinder,
          RevRec, Tie, WRC, xFinder]


def test_recommendation_metrics(models_cls, path=None, data_args=None, filter_args=None):
    if filter_args is None:
        filter_args = {}
    if data_args is None:
        data_args = {}
    if path is None:
        path = 'results/recmetrics'

    for mdl_cls in models_cls:
        setup = model_setup[mdl_cls.__name__]
        dataset_kwargs = setup['dataset_kwargs'].extend(data_args)

        dataset = dataset_classes[setup['dataset']](**dataset_kwargs)
        data_iterator = iterator_classes[setup['iterator']](**setup['iterator_args'])

        model_kwargs = setup['model_kwargs'].extend(filter_args)
        if setup['items2id']:
            model = mdl_cls(dataset.get_items2ids, **model_kwargs)
        else:
            model = mdl_cls(**model_kwargs)

        tester = RecTester()
        res = tester.test_recommender(model, data_iterator)
        print(f'Finished for {mdl_cls.__name__}')
        save_results(path, res[0], model)


if __name__ == '__main__':
    test_recommendation_metrics(models)
