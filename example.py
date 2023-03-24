from batcore.data import PullLoader
from batcore.tester import RecTester
from batcore.data import MRLoaderData
from batcore.baselines import *
from batcore.data import get_gerrit_dataset
import pandas as pd
from datetime import datetime

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    # loads data of OpenStack from ./raw_dataset/review.opendev.org folder with events from 01.12.2022 to 01.01.2023.
    # Accounts with close names are matched together and encoded to the same id

    # data = MRLoaderData('./raw_dataset/review.opendev.org',
    #                     from_date=datetime(year=2022, month=12, day=1),
    #                     to_date=datetime(year=2023, month=1, day=1),
    #                     )
    # data.to_checkpoint('./other_data/openstack_alias')

    # reloads saved data from the checkpoint
    data = MRLoaderData().from_checkpoint('./other_data/openstack_alias')
    
    # gets dataset for the CN model. Pull request with more than 56 files are removed
    dataset = get_gerrit_dataset(data, max_file=56, model_cls=RevRec)

    # creates an iterator over dataset that iterates over pull request one-by-one
    data_iterator = PullLoader(dataset, 10)

    # create a CN model. dataset.get_items2ids() provides model with necessary encodings (eg. users2id, files2id) for
    # optimization of evaluation
    model = RevRec(dataset.get_items2ids())

    # create a tester object
    tester = RecTester()

    # run the tester and receive dict with all the metrics
    res = tester.test_recommender(model, data_iterator)

    print(res[0])
