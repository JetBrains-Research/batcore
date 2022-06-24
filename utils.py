import os

import numpy as np
import pandas as pd


def get_df(path):
    dfs = {}
    for df in os.listdir(path):
        try:
            dfs[df.split('.')[0]] = pd.read_csv(path + f'/{df}', sep='|')
        except:
            continue
    return dfs


def count_confidence(sample):
    # for 0.95 confidence interval
    p = np.mean(sample)
    n = len(sample)
    z = 1.96
    if n * p > 10:
        # De Moivre–Laplace theorem
        d = z * np.sqrt(p * (1 - p) / len(sample))
    else:
        # Poisson Limit theorem
        d = z * np.sqrt(p / n / n)

    return d


def count_metrics(res):
    res = res.copy()

    rrs = []
    for _, row in res.iterrows():
        rr = [np.inf]
        for t in row['rev']:
            rr = min(rr, 1 + np.where(np.array(row['top-10']) == t)[0])
        rrs.append(1 / rr[0])
    res['rr'] = rrs

    res['rev'] = res['rev'].apply(set)
    for i in [1, 3, 5, 10]:
        res[f'top-{i}'] = (res['rev'] - res[f'top-{i}'].apply(set)).apply(lambda x: len(x))

    res['rev'] = res['rev'].apply(lambda x: len(x))

    for i in [1, 3, 5, 10]:
        res[f'top-{i}'] = res[f'top-{i}'] < res['rev']

    res.drop('rev', axis=1)

    res_dict = {'rr': res['rr'].mean()}

    for i in [1, 3, 5, 10]:
        res_dict[f'top-{i}'] = (res[f'top-{i}'].mean(), count_confidence(res[f'top-{i}'].to_numpy()))
    return res_dict
