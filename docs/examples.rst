.. _examples_toplevel:

========
Examples
========

Running implemented models
==========================

.. code-block:: python

    from batcore.data import PullLoader
    from batcore.tester import RecTester
    from batcore.data import MRLoaderData
    from batcore.baselines import CN
    from batcore.data import get_gerrit_dataset

    # reloads saved data from the checkpoint
    data = MRLoaderData().from_checkpoint('projects/openstack')

    # gets dataset for the CN model. Pull request with more than 56 files are removed
    dataset = get_gerrit_dataset(data, max_file=56, model_cls=CN)

    # creates an iterator over dataset that iterates over pull request one-by-one
    data_iterator = PullLoader(dataset, 10)

    # creates a CN model. dataset.get_items2ids() provides model with necessary encodings
    # (eg. users2id, files2id) for optimization of evaluation
    model = CN(dataset.get_items2ids())

    # create a tester object
    tester = RecTester()

    # run the tester and receive dict with all the metrics
    res = tester.test_recommender(model, data_iterator)


Loading dataset from MRLoader output
====================================

.. code-block:: python

    from batcore.data import MRLoaderData

    data = MRLoaderData('path', # path to the directory containing output of MRLoader
                         bots='', # path to file with bots or 'auto'
                         project_name='', # name of the project for in case of auto bot detection
                         from_checkpoint=False, # when true reloads saves data
                         from_date=datetime(), # all events before are removed
                         to_date=datetime(), # all events after are removed
                         factorize_users=True, # when true users are replaced withs numerical ids
                         alias=True, # when true users with close names/emails/logins are treated as one
                         remove_bots=True # when true bots are removed from the data
                        )

Creating Dataset from GerritLoader
==================================

.. code-block:: python

    from batcore.data import StandardDataset

    dataset = StandardDataset(data, # instance of MRLoaderData
                              max_file=100, # number of maximum files in a pull request
                              commits=False, # if true commits are included in the dataset
                              comments=False, # if true comments are included in the dataset
                              user_items=False, # if true makes a user2id map
                              file_items=False, # if true makes a file2id map
                              pull_items=False, # if true makes a pull2id map
                              remove_empty=False, # if true pull requests w/out reviewers are removed
                              owner_policy='', # strategy for identification of the author of the pull request
                              remove=[] # list of columns/features that will be removed
                             )

Creating new model
======================

To create new mode one can simply implement abstract class RecommenderBase with `fit` and `predict` methods.

* `fit` is a methods that trains the model on the given `data`
* `predict` is methods that returns list of candidates to review give pull request `pull`

For any model implementing those two methods testing can be done the same way as with implemented baselines. The only exception is changing dataset initialization from `get_gerrit_dataset` to manual initialization. A simple example of the recommender implementation can be found below:

.. code-block:: python

    from batcore.modelbase import RecommenderBase
    import numpy as np

    class SimpleRecommender(RecommenderBase):
        def __init__(self):
            super().__init__()
            self.reviewers = []

        def predict(self, pull, n=10):
            return [np.random.choice(self.reviewers)]

        def fit(self, data):
            self.reviewers.extend(event['reviewer'])
