from datetime import timedelta

from utils import *


# def get_data(dataset_name):
#     if dataset_name in ['beam', 'flink', 'kafka', 'spark', 'zookeeper']:
#         return get_df(dataset_name)


def run_on_history(model, events):
    from_date = events.date.min()
    end_date = events.date.max()

    to_date = timedelta(440, 0) + from_date
    init = events[(events.date < to_date) & (events.date > from_date)]

    model.initialize(init)

    preds = []
    targets = []

    # for i in tqdm(range(10000)):
    i = 0
    while from_date <= end_date:
        from_date = to_date
        to_date = from_date + timedelta(7, 0)

        cur_events = events[(events.date < to_date) & (events.date > from_date)]

        pred, target = model.process(cur_events)
        preds += pred
        targets += target
        # i += 1
        # if i > 5:
        #     break
    return preds, targets


def calculate_metrics_on_review(pred, target):
    res = np.zeros(5)
    for i, k in enumerate([1, 3, 5, 10]):
        if len(set(pred[:k]).intersection(set(target))):
            res[i] += 1
    rr = np.inf
    for t in target:
        rr = min(rr, 1 + np.where(pred == t)[0])
    res[4] = 1 / rr
    return res


def calculate_metrics(results):
    res = np.zeros(5)
    cnt = 0
    for pred, target in zip(*results):
        if len(target) > 0:
            res += calculate_metrics_on_review(pred, target)
            cnt += 1
    return res / cnt


def run_experiments(model_class, hyperparameters=None, datasets=None):
    if hyperparameters is None:
        hyperparameters = {}
    if datasets is None:
        datasets = ['beam', 'flink', 'kafka', 'spark', 'zookeeper', 'libreoffice', 'android', 'qt', 'openstack']

    res = {}
    for d in datasets:
        data, files, users = get_data(d)
        model = model_class(files, users, **hyperparameters)
        results = run_on_history(model, data)
        metrics = calculate_metrics(results)
        res[d] = metrics

    return res
