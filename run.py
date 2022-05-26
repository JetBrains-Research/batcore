from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from recsys.estimator import train_als
from recsys.recommender import recommend
from recsys.utils import get_data, count_metrics

path = 'zookeeper/2020-01-10'

pulls, commits, mapping_user_rev, mapping_user_com, mapping_file_rev, mapping_file_com = get_data(path)

from_date = commits.date.min()
end_date = commits.date.max()

to_date = timedelta(2700, 0) + from_date
train_com = commits[(commits.date < to_date) & (commits.date >= from_date)]
train_com = train_com.groupby(['file_path', 'author_login']).count().reset_index().rename(
    {'date': 'number', 'author_login': 'login'}, axis=1)
train_rev = pulls[(pulls.date < to_date) & (pulls.date >= from_date)]
train_rev = train_rev.groupby(['file_path', 'reviewer_login', 'author_login']).count().reset_index().drop(
    ['date', 'author_login'], axis=1).rename({'reviewer_login': 'login'}, axis=1)

res = None
col_item, col_user, _ = ['file_path', 'login', 'number']

hyperparameters = {'transform': {'alpha': 30},
                   'als': {'factors': 50,
                           'iterations': 30,
                           'regularization': 100},
                   'gamma': 0.999}

for i in range(10):
    print(f"[{i}] - started test df construction")
    test_date = to_date + timedelta(7, 0)
    test_rev = pulls[(pulls.date >= to_date) & (pulls.date < test_date)]
    test_rec = test_rev.groupby('number')[['file_path', 'reviewer_login']].agg(
        {'file_path': list, 'reviewer_login': lambda x: list(set(x))}).reset_index()

    shape = (len(mapping_user_com.id2item), len(mapping_file_com.id2item))

    print(f"[{i}] - started als traing")
    model_commits_als, matcom = train_als(train_com)
    model_reviews_als, matrev = train_als(train_rev)
    mapping_user_com.set_mask(train_com, 'login')
    mapping_user_rev.set_mask(train_rev, 'login')

    mapping_file_com.set_mask(train_com, 'file_path')
    mapping_file_rev.set_mask(train_rev, 'file_path')

    print(f"[{i}] - started recommending")
    res_cur = recommend(test_rec,
                        model_reviews_als,
                        model_commits_als,
                        mapping_user_rev,
                        mapping_user_com,
                        mapping_file_rev,
                        [1, 3, 5, 10])

    if res is None:
        res = res_cur
    else:
        res = pd.concat([res, res_cur], axis=0)

    print(f"[{i}] - started train construction")
    train_com2 = commits[(commits.date >= to_date) & (commits.date < test_date)]
    train_com2 = train_com2.groupby(['file_path', 'author_login']).count().reset_index().rename(
        {'date': 'number', 'author_login': 'login'}, axis=1)
    train_rev2 = pulls[(pulls.date >= to_date) & (pulls.date < test_date)]
    train_rev2 = train_rev2.groupby(['file_path', 'reviewer_login']).count().reset_index().drop('date', axis=1).rename(
        {'reviewer_login': 'login'}, axis=1)

    train_rev['number'] = train_rev['number'] * hyperparameters['gamma']
    train_com['number'] = train_com['number'] * hyperparameters['gamma']

    train_rev = pd.concat([train_rev2, train_rev], axis=0).groupby(['file_path', 'login']).sum().reset_index()
    train_com = pd.concat([train_com2, train_com], axis=0).groupby(['file_path', 'login']).sum().reset_index()

    to_date = test_date
    if to_date > end_date:
        break

print(count_metrics(res))
