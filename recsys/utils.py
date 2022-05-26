import os

import pandas as pd

from recsys.mapping import *


def get_df(path):
    dfs = {}
    for df in os.listdir(path):
        try:
            dfs[df.split('.')[0]] = pd.read_csv(path + f'/{df}', sep='|')
        except:
            continue
    return dfs


def get_data(path):
    dfs = get_df(path)
    pulls = dfs['pull_file'].merge(dfs['reviewer']).merge(dfs['pull'][['number', 'created_at']], left_on='pull_number',
                                                          right_on='number').merge(dfs['pull_author']).dropna()
    pulls['date'] = pd.to_datetime(pulls.created_at).dt.tz_localize(None)
    commits = dfs['commit_file'].merge(dfs['commit_author']).merge(dfs['commit'][['oid', 'committed_date']],
                                                                   left_on='oid', right_on='oid').dropna()
    commits['date'] = pd.to_datetime(commits.committed_date).dt.tz_localize(None)

    pulls = pulls.drop(['pull_number', 'created_at'], axis=1)
    commits = commits.drop(['oid', 'file_oid', 'committed_date'], axis=1)

    #     pulls = pulls.sort_values('date')
    #     commits = commits.sort_values('date')

    id2file = np.unique(np.hstack((pulls.file_path, commits.file_path)))
    id2user = np.unique(np.hstack((pulls.reviewer_login, pulls.author_login, commits.author_login)))

    file2id = {f: i for i, f in enumerate(id2file)}
    user2id = {u: i for i, u in enumerate(id2user)}

    pulls['file_path'] = pulls.file_path.apply(lambda x: file2id[x])
    pulls['reviewer_login'] = pulls.reviewer_login.apply(lambda x: user2id[x])
    pulls['author_login'] = pulls.author_login.apply(lambda x: user2id[x])

    commits['file_path'] = commits.file_path.apply(lambda x: file2id[x])
    commits['author_login'] = commits.author_login.apply(lambda x: user2id[x])

    # TODO make one class with 2 masks
    commit_file_mapping = MappingWithFallback(file2id, id2file)
    pull_file_mapping = MappingWithFallback(file2id, id2file)

    commit_user_mapping = Mapping(file2id, id2file)
    pull_user_mapping = Mapping(file2id, id2file)

    return pulls, commits, pull_user_mapping, commit_user_mapping, pull_file_mapping, commit_file_mapping


def count_metrics(res):
    res = res.copy()

    rrs = []
    for _, row in res.iterrows():
        rr = [np.inf]
        for t in row['rev']:
            rr = min(rr, 1 + np.where(row['top-10'] == t)[0])
        #     break
        rrs.append(1 / rr[0])
    res['rr'] = rrs

    res['rev'] = res['rev'].apply(set)
    for i in [1, 3, 5, 10]:
        res[f'top-{i}'] = (res['rev'] - res[f'top-{i}'].apply(set)).apply(lambda x: len(x))

    res['rev'] = res['rev'].apply(lambda x: len(x))

    for i in [1, 3, 5, 10]:
        res[f'top-{i}'] = res[f'top-{i}'] < res['rev']

    res.drop('rev', axis=1)
    return res.mean()
