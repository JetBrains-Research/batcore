import warnings
from itertools import product

from Recommender import *
from pipeline import *

warnings.filterwarnings('ignore')

# import argparse


if __name__ == '__main__':
    print('start experiments')
    names = ['a@1', 'a@3', 'a@5', 'a@10', 'mrr']

    data, files, users = get_data('beam', True)

    hyperparams = {'eps': [0.001], 'd': [128], 'alpha': [1], 'lam': [0.1]}
    # hyperparams = {'eps': [0.001, 0.01], 'd': [128, 256], 'alpha': [1], 'lam': [0.1, 1]}

    keys = hyperparams.keys()

    for params in product(*hyperparams.values()):
        param_dict = {arg: val for (arg, val) in zip(keys, params)}
        print(f'Started test for {param_dict}\n')
        model = SparseRecommender(files, users, **param_dict)

        results = run_on_history(model, data)
        metrics = calculate_metrics(results)
        print(f'Results:\n')
        for n, m in zip(names, metrics):
            print(f"{n}:{m}\n", flush=True)
