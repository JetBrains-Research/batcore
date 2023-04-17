from copy import deepcopy

from batcore.data import StandardDataset, RevRecDataset, TieDataset, RevFinderDataset
from batcore.baselines import *

default_args = {'max_file': 50,
                'commits': False,
                'comments': False,
                'user_items': False,
                'file_items': False,
                'pull_items': False,
                'owner_policy': 'author_owner_fallback',
                'remove': ['owner']}


def remove_nones(kwargs):
    keys_to_remove = []
    for key in kwargs:
        if kwargs[key] is None:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del kwargs[key]


def get_gerrit_dataset(
        dataset,
        max_file=None,
        commits=None,
        comments=None,
        user_items=None,
        file_items=None,
        pull_items=None,
        owner_policy=None,
        remove=None,
        model_cls=None):
    """
    returns a dataset object with specified attributes or default dataset for the specified model
    :param dataset: GerritLoader-like object
    :param max_file: maximum number of files that a review can have
    :param commits: if False commits are omitted from the data
    :param comments: if False comments are omitted from the data
    :param user_items: if True user2id map is created
    :param file_items: if True file2id map is created
    :param pull_items: if true pull2id map is created
    :param owner_policy: how pull owners are calculated.
        * None - owners are unchanged
        * author - commit authors of the pull are treated as owners
        * author_no_na - commit authors of the pull are treated as owners. pulls without an author are removed
        * author_owner_fallback - if pull has author, owner field set to the author. Otherwise, nothing is done
    :param remove: list of columns to remove from the reviewers. Can be a subset of ['owner', 'author']
    :param model_cls: class implementing RecommenderBase interface or None. When class is specified suitable Dataset
    will be returned
    """
    kwargs = {'max_file': max_file,
              'commits': commits,
              'comments': comments,
              'user_items': user_items,
              'file_items': file_items,
              'pull_items': pull_items,
              'owner_policy': owner_policy,
              'remove': remove}

    remove_nones(kwargs)
    data_args = deepcopy(default_args)
    data_args.update(kwargs)

    if issubclass(model_cls, RevFinder):
        return RevFinderDataset(dataset, **data_args)
    elif issubclass(model_cls, ACRec):
        data_args['comments'] = True
        return StandardDataset(dataset, **data_args)
    elif issubclass(model_cls, CN):
        data_args['comments'] = True
        data_args['user_items'] = True
        return StandardDataset(dataset, **data_args)
    elif issubclass(model_cls, RevRec):
        data_args['comments'] = True
        data_args['user_items'] = True
        return RevRecDataset(dataset, **data_args)
    elif issubclass(model_cls, Tie):
        return TieDataset(dataset, **data_args)
    elif issubclass(model_cls, WRC):
        data_args['user_items'] = True
        data_args['file_items'] = True
        return StandardDataset(dataset, **data_args)
    elif issubclass(model_cls, cHRev):
        data_args['comments'] = True
        return StandardDataset(dataset, **data_args)
    elif issubclass(model_cls, xFinder):
        data_args['commits'] = True
        return StandardDataset(dataset, **data_args)
    else:
        return StandardDataset(dataset, **data_args)
