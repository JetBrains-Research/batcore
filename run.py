import warnings
from Recommender import *
from pipeline import *

warnings.filterwarnings('ignore')

# import argparse


if __name__ == '__main__':
    print('start experiments')
    metrics = run_experiments(SparseRecommender, None, ['beam'])
    names = ['a@1', 'a@3', 'a@5', 'a@10', 'mrr']
    # with open('results/results.txt', 'w') as f:
    #     for n, m in zip(names, metrics['beam']):
    #         f.write(f"{n}:{m}\n")
    #
    for n, m in zip(names, metrics['beam']):
        print(f"{n}:{m}\n")
