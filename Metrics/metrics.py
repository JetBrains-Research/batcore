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


def count_mean(metric_vals, bootstrap_prob=0.5, bootstrap_repeat=100):
    metric_vals = np.array(metric_vals)
    real_mean = np.mean(metric_vals)

    subsample_ind = np.random.rand(len(metric_vals), bootstrap_repeat) < bootstrap_prob
    subsample = subsample_ind * metric_vals.reshape(-1, 1)
    subsample_cnt = subsample_ind.sum(axis=0)

    bootstrap_means = subsample.sum(axis=0) / subsample_cnt
    bt_mean = np.mean(bootstrap_means)
    bt_var = np.var(bootstrap_means, ddof=1)

    return real_mean, bt_var


def count_mrr(res):
    rrs = []
    for _, row in res.iterrows():
        rr = [np.inf]
        for t in row['rev']:
            rr = min(rr, 1 + np.where(np.array(row['top-10']) == t)[0])
        rrs.append(1 / rr[0])

    return {'mrr': count_mean(rrs)}


def recall(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred)) / len(gt)


def precision(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred)) / len(pred)


def f1score(gt, pred):
    recall_score = recall(gt, pred)
    precision_score = precision(gt, pred)
    return 2 * precision_score * recall_score / (precision_score + recall_score + 1e-8)


def accuracy(gt, pred):
    gt = set(gt)
    pred = set(pred)

    return len(gt.intersection(pred)) > 0


def count_topk_metric(res, top_k, func, name='metric'):
    result = {}
    for k in top_k:
        inter_result = res.apply(lambda x: func(x['rev'], x[f'top-{k}']), axis=1)
        result[f'{name}@{k}'] = count_mean(inter_result)

    return result


metric_func = {'acc': accuracy,
               'rec': recall,
               'prec': precision,
               'f1': f1score}


def count_metrics(res, metrics=None, top_k=None):
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
