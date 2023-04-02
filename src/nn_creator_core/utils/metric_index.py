from solver.master.utils.accuracy_metric import get_performance
from solver.master.utils.metrics import acc

METRICS = {
    "range_scaled_accuracy": get_performance,
    "range_scaled_acc": get_performance,
    "performance_accuracy": acc,
    "performance_acc": acc,
}


def calc_metrics(data, pred, metrics, **kwargs):
    results = {}
    for metric in metrics:
        if callable(metric):
            results[metric.__name__] = metric(data, pred, **kwargs)
        elif type(metric) == str:
            f = METRICS[metric]
            results[metric] = f(data, pred, **kwargs)
        else:
            raise ValueError()
    return results
