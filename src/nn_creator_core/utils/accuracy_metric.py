import pandas as pd
import numpy as np
from typing import Union

from nn_creator_core.utils.utils import default_kwargs
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))


def mm_scaled_error(true, pred, low, high):
    diff = (true - pred).abs()
    r = (high - low).abs()
    a = diff / r
    return a


def minimal_nz_error(true, pred, nz):
    pass


def error2acc(errors: pd.DataFrame, t, mode="per_column"):
    if mode == "per_column":
        acc = (errors <= t).sum() / errors.shape[0]
        return acc
    elif mode == "per_row":
        flags = (errors <= t).apply(np.all, axis=1)
        acc = flags.sum() / flags.shape[0]
        return acc
    else:
        raise NotImplementedError()


@default_kwargs(threshold=0.03, measure="rmse", targets=None, ranges=None)
def get_performance(true: pd.DataFrame,
                    pred: pd.DataFrame,
                    **kwargs):
    results = {}
    target_list = kwargs['targets']
    ranges = kwargs['ranges']
    threshold = kwargs['threshold']
    measure = kwargs["measure"]

    f = mean_squared_error if measure == "mse" else rmse
    if target_list:
        target_true_df = true.copy()[target_list]
        target_pred_df = pred.copy()[target_list]
    else:
        target_true_df = true.copy()
        target_pred_df = pred.copy()

    target_true_df = target_true_df.reset_index(drop=True)
    target_pred_df = target_pred_df.reset_index(drop=True)

    overall_measure_value = f(target_true_df.values, target_pred_df.values)
    feature_measure_values = pd.Series([f(true[c], pred[c]) for c in target_pred_df.columns],
                                       index=target_pred_df.columns)
    results["overall_measure_value"] = overall_measure_value
    results["feature_measure_values"] = feature_measure_values

    if ranges:
        low, high = ranges
        low = low[target_list] if target_list else low
        high = high[target_list] if target_list else high
        errors = mm_scaled_error(target_true_df, target_pred_df, low, high)
        results["mm_scaled_mean_percentage_error"] = errors.mean()
        results["mm_per_column_acc"] = error2acc(errors, threshold, "per_column")
        results["mm_per_row_acc"] = error2acc(errors, threshold, "per_row")
        results["mean_mm_per_column_acc"] = results["mm_per_column_acc"].mean()

    return results


def plot_perf(tables, targets):
    true_train, pred_train, true_val, pred_val, true_test, pred_test = tables

    low = true_train[targets].min()
    high = true_train[targets].max()
    train_perf = get_performance(true_train, pred_train, ranges=[low, high], target_list=targets)
    val_perf = get_performance(true_val, pred_val, ranges=[low, high], target_list=targets)
    test_perf = get_performance(true_test, pred_test, ranges=[low, high], target_list=targets)

    train_mean = np.mean(train_perf["mm_per_column_acc"])
    val_mean = np.mean(val_perf["mm_per_column_acc"])
    test_mean = np.mean(test_perf["mm_per_column_acc"])

    fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    ax1 = axes  # .flatten()

    acc_df = pd.concat([train_perf['mm_per_column_acc'],
                        val_perf['mm_per_column_acc'],
                        test_perf['mm_per_column_acc']], axis=1, names=["train", "val", "test"])
    acc_df.columns = ["train", "val", "test"]
    acc_df.plot.bar(ax=ax1, legend=False)
    ax1.legend(labels=["train ({} in mean)".format(train_mean),
                       "val ({} in mean)".format(val_mean),
                       "test ({} in mean)".format(test_mean)],
               bbox_to_anchor=(1., 1))
    ax1.grid()
    plt.tight_layout()
    plt.show(block=True)


def plot_performance(performances, train_histories=None, trues=None, preds=None, labels=("train", "val")):
    total_means = []
    accuracies = []
    for performance, label in zip(performances, labels):
        total_means.append(np.mean(performance["mm_per_column_acc"]))
        accuracies.append(performance['mm_per_column_acc'])

    fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    ax1 = axes  # .flatten()

    # pca = PCA(n_components=1)
    # df = trues[0]
    # pca.fit(df.values)
    # t = pca.transform(trues[0].values)
    # p = pca.transform(preds[0].values)
    #
    # ax1.scatter(t, p)
    # ax1.grid()
    # ax1.set_xlabel("True")
    # ax1.set_ylabel("Pred")

    acc_df = pd.concat(accuracies, axis=1)
    acc_df.columns = labels
    acc_df.plot.bar(ax=ax1, legend=False)
    l = ["{} ({} in mean)".format(name, means) for name, means in zip(labels, np.round(total_means, 2))]
    ax1.legend(labels=l,
               bbox_to_anchor=(1., 1))
    ax1.grid()

    plt.tight_layout()
    plt.savefig('data/Saved/perf.png')
    plt.show(block=True)

