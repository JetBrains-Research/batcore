import numpy as np


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


def bootstrap_estimation(metric_vals, bootstrap_size=50, bootstrap_repeat=1000):
    """
    :param metric_vals: metrics values per data-point
    :param bootstrap_prob: probability of the data-point to appear in sub-sample
    :param bootstrap_repeat: number of bootstrap iterations
    :return: real mean and bootstrap variance estimation
    """

    if bootstrap_size is None:
        bootstrap_size = len(metric_vals)

    metric_vals = np.array(metric_vals)

    real_mean = np.mean(metric_vals)

    subsample = np.random.choice(metric_vals,
                                 size=bootstrap_size * bootstrap_repeat,
                                 replace=True).reshape(bootstrap_size, -1)

    bootstrap_means = subsample.sum(axis=0) / bootstrap_size

    bt_mean = np.mean(bootstrap_means)
    bt_var = np.var(bootstrap_means, ddof=1)

    return real_mean, bt_var


def count_mrr(res):
    """
    :param res: pd.DataFrame with prediction done by the model. Should have column 'rev' that represents ground truth.
    Also column 'top-10' that represents best 10 suggestions
    :return: mean and std of reciprocal ranks
    """
    rrs = []
    for _, row in res.iterrows():
        rr = [np.inf]
        for t in row['rev']:
            rr = min(rr, 1 + np.where(np.array(row['top-10']) == t)[0])
        rrs.append(1 / rr[0])

    return {'mrr': bootstrap_estimation(rrs)}


def recall(gt, pred):
    """
    :param gt: ground truth
    :param pred: predictions
    :return: Recall score
    """
    gt = set(gt)
    pred = set(pred)
    if len(gt):
        return len(gt.intersection(pred)) / len(gt)
    return 0


def precision(gt, pred):
    """
    :param gt: ground truth
    :param pred: predictions
    :return: precision score
    """
    gt = set(gt)
    pred = set(pred)
    if len(pred):
        return len(gt.intersection(pred)) / len(pred)
    return 0


def f1score(gt, pred):
    """
    :param gt: ground truth
    :param pred: predictions
    :return: F1 score
    """
    recall_score = recall(gt, pred)
    precision_score = precision(gt, pred)
    return 2 * precision_score * recall_score / (precision_score + recall_score + 1e-8)


def accuracy(gt, pred):
    """
    :param gt: ground truth
    :param pred: predictions
    :return: accuracy score
    """
    gt = set(gt)
    pred = set(pred)

    return len(gt.intersection(pred)) > 0


def count_topk_metric(res, top_k, metric, name='metric'):
    """
    :param res: pd.DataFrame with prediction done by the model. Column 'rev' represents ground truth. Column 'top-k'
    represents best k suggestions
    :param top_k: list with amount of the best suggestions
    :param metric: metric function
    :param name: name of the metric
    :return: dict with mean values and stds for each metric calculated for each of the top-k suggestions
    """
    result = {}
    for k in top_k:
        inter_result = res.apply(lambda x: metric(x['rev'], x[f'top-{k}']), axis=1)
        result[f'{name}@{k}'] = bootstrap_estimation(inter_result)

    return result


metric_func = {'acc': accuracy,
               'rec': recall,
               'prec': precision,
               'f1': f1score}


def count_metrics(res, metrics=None, top_k=None):
    """
    :param res: pd.DataFrame with prediction done by the model.
    :param metrics: metrics to calculate
    :param top_k: list of k-s for top k metrics
    :return:  dict with mean values and variance for each metric
    """
    if top_k is None:
        top_k = [1, 3, 5, 10]

    if metrics is None:
        metrics = ['acc', 'mrr', 'rec', 'prec', 'f1']
    res = res.copy()
    result = {}
    for metric in metrics:
        if metric == 'mrr':
            result.update(count_mrr(res))
        else:
            result.update(count_topk_metric(res, top_k, metric_func[metric], metric))
    return result
