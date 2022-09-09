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


def count_accuracy(res, top_k):
    num_rec = res['rev'].apply(lambda x: len(x))
    result = {}
    rev_set = res['rev'].apply(set)
    for k in top_k:
        res[f'acc@{k}'] = (rev_set - res[f'top-{k}'].apply(set)).apply(lambda x: len(x))
        res[f'acc@{k}'] = res[f'acc@{k}'] < num_rec
        result[f'acc@{k}'] = count_mean(res[f'acc@{k}'])

    return result


def count_mrr(res):
    rrs = []
    for _, row in res.iterrows():
        rr = [np.inf]
        for t in row['rev']:
            rr = min(rr, 1 + np.where(np.array(row['top-10']) == t)[0])
        rrs.append(1 / rr[0])

    return {'mrr': count_mean(rrs)}


def recall_one(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred)) / len(gt)


def precision_one(gt, pred):
    gt = set(gt)
    pred = set(pred)
    return len(gt.intersection(pred)) / len(pred)


def f1score_one(gt, pred):
    recall = recall_one(gt, pred)
    precision = precision_one(gt, pred)
    return 2 * precision * recall / (precision + recall + 1e-8)


def count_recall(res, top_k):
    result = {}
    for k in top_k:
        res[f'rec@{k}'] = res.apply(lambda x: recall_one(x['rev'], x[f'top-{k}']), axis=1)
        result[f'rec@{k}'] = count_mean(res[f'rec@{k}'])

    return result


def count_precision(res, top_k):
    result = {}
    for k in top_k:
        res[f'prec@{k}'] = res.apply(lambda x: precision_one(x['rev'], x[f'top-{k}']), axis=1)
        result[f'prec@{k}'] = count_mean(res[f'prec@{k}'])

    return result


def count_f1(res, top_k):
    result = {}
    for k in top_k:
        res[f'f1@{k}'] = res.apply(lambda x: f1score_one(x['rev'], x[f'top-{k}']), axis=1)
        result[f'f1@{k}'] = count_mean(res[f'f1@{k}'])

    return result


def count_metrics(res, top_k):
    res = res.copy()
    result = {}
    result.update(count_accuracy(res, top_k))
    result.update(count_mrr(res))
    result.update(count_recall(res, top_k))
    result.update(count_precision(res, top_k))
    result.update(count_f1(res, top_k))
    return result
