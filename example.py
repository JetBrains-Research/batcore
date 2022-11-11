from baselines import *
from data import *
from data import StreamDataLoader
from tester import RecTester
import pandas as pd

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    # loads data of OpenStack from dataset-7/review.openstack.org folder with events from 01.07.2011 to 31.05.2012.
    # All accounts in projects/openstack/bots.csv are treated as bots and removed.
    # Accounts with close names are matched together and encoded to the same id

    # data = GerritLoader('dataset-7/review.openstack.org',
    #                     bots='projects/openstack/bots.csv',
    #                     project_name='OpenStack',
    #                     from_checkpoint=False,
    #                     from_date=datetime(year=2011, month=7, day=1),
    #                     to_date=datetime(year=2012, month=5, day=31),
    #                     factorize_users=True, alias=True,
    #                     remove_bots=True)

    # reloads saved data from the checkpoint
    data = GerritLoader('projects/openstack', from_checkpoint=True)

    # gets dataset for the CN model. Pull request with more than 56 files are removed
    dataset = get_gerrit_dataset(data, max_file=56, model_cls=CN)

    # creates an iterator over dataset that iterates over pull request one-by-one
    data_iterator = StreamDataLoader(dataset, 1)

    # create a CN model. dataset.get_items2ids() provides model with necessary encodings (eg. users2id, files2id) for
    # optimization of evaluation
    model = CN(dataset.get_items2ids())

    # create a tester object
    tester = RecTester()

    # run the tester and receive dict with all the metrics
    res = tester.test_recommender(model, data_iterator)

    print(res[0])
