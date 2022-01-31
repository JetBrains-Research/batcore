import os

import numpy as np
import pandas as pd

from Container import ObjectContainer

MISSING_ID = -1


def get_reviewers(event, additional_data):
    return additional_data['reviewer'][additional_data['reviewer'].pull_number == event.id].reviewer_login.to_list()


def reshape(matrix, new_shape):
    assert matrix.shape[0] <= new_shape[0]
    assert matrix.shape[1] <= new_shape[1]

    new_matrix = np.zeros(new_shape)
    new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix

    return new_matrix


def get_df(pr):
    dfs = {}
    # for df in os.listdir(f'github_csv/csv/github/apache/{pr}/2020-01-10'):
    #     dfs[df.split('.')[0]] = pd.read_csv(f'github_csv/csv/github/apache/{pr}/2020-01-10/{df}', sep='|')
    for df in os.listdir(f'beam/csv/github/apache/{pr}/2020-01-10'):
            dfs[df.split('.')[0]] = pd.read_csv(f'beam/csv/github/apache/{pr}/2020-01-10/{df}', sep='|')

    return dfs


def get_data(project):
    dfs = get_df(project)

    pull_events = dfs['pull'][['created_at', 'number']]
    commit_events = dfs['commit'][['committed_date', 'oid']]

    commit_events = commit_events.rename(columns={'committed_date': 'date', 'oid': 'id'})
    pull_events = pull_events.rename(columns={'created_at': 'date', 'number': 'id'})

    pull_events['type'] = 'pull'
    commit_events['type'] = 'commit'

    events = pd.concat([pull_events, commit_events])
    events.date = pd.to_datetime(events.date).dt.tz_localize(None)
    events = events.sort_values('date').reset_index().drop('index', axis=1)

    events = events.merge(
        events.merge(dfs['commit_file'], left_on='id', right_on='oid', how='outer').groupby('id')['file_path'].apply(
            list), on='id')
    events = events.merge(events.merge(dfs['pull_file'], left_on='id', right_on='pull_number', how='outer',
                                       suffixes=['', '_pull']).groupby('id')['file_path_pull'].apply(list), on='id')
    events.loc[events.type == 'pull', 'file_path'] = events[events.type == 'pull'].file_path_pull
    events = events.drop(['file_path_pull'], axis=1)

    events = events.merge(events.merge(dfs['commit_author'], left_on='id', right_on='oid', how='outer').groupby('id')[
                              'author_login'].apply(list), on='id')
    events = events.merge(events.merge(dfs['pull_author'], left_on='id', right_on='pull_number', how='outer',
                                       suffixes=['', '_pull']).groupby('id')['author_login_pull'].apply(list), on='id')
    events.loc[events.type == 'pull', 'author_login'] = events[events.type == 'pull'].author_login_pull
    events = events.drop(['author_login_pull'], axis=1)

    events = events.merge(
        events.merge(dfs['reviewer'], left_on='id', right_on='pull_number', how='outer').groupby('id')[
            'reviewer_login'].apply(list), on='id')

    events = events.merge(
        events.merge(dfs['participant'], left_on='id', right_on='pull_number', how='outer').groupby('id')[
            'participant_login'].apply(list), on='id')

    events.participant_login = events['participant_login'].apply(set) - events['reviewer_login'].apply(set)
    events.reviewer_login = events.reviewer_login.apply(set) - set([np.nan])
    events.author_login = events.author_login.apply(set) - set([np.nan])
    events.file_path = events.file_path.apply(set) - set([np.nan])

    files = set.union(*events.file_path)

    authors = set.union(*events.author_login)
    review = set.union(*events.reviewer_login)
    part = set.union(*events.participant_login)

    users = authors.union(review).union(part)

    files = ObjectContainer(files)
    users = ObjectContainer(users)

    events.file_path = events.file_path.apply(lambda x: files.get_ids(x))
    events.author_login = events.author_login.apply(lambda x: users.get_ids(x))
    events.reviewer_login = events.reviewer_login.apply(lambda x: users.get_ids(x))
    events.participant_login = events.participant_login.apply(lambda x: users.get_ids(x))

    return events, files, users
