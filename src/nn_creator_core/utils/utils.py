import functools
import shutil

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import Union

import zipfile
import os
import io


def dir2bytes(dir_path: str):
    shutil.make_archive(dir_path, 'zip', dir_path)
    with open("{}.zip".format(dir_path), "rb") as f:
        archive_content = f.read()
    os.remove("{}.zip".format(dir_path))
    return archive_content


def bytes2dir(archive_content: bytes, dir_path):
    with io.BytesIO(archive_content) as archive_data:
        with zipfile.ZipFile(archive_data, 'r') as zip_file:
            zip_file.extractall(dir_path)


def default_kwargs(**defaultKwargs):
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            defaultKwargs.update(kwargs)
            return fn(*args, **defaultKwargs)

        return g

    return actual_decorator


def kwargs_defaults(**default_kwargs_values):
    # chatGPT
    def actual_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            combined_kwargs = {**default_kwargs_values, **kwargs}
            if len(combined_kwargs) > len(default_kwargs_values):
                duplicated_keys = set(default_kwargs_values) & set(kwargs)
                raise ValueError(f"duplicate kwargs keys: {duplicated_keys}")
            return fn(*args, **combined_kwargs)

        return wrapper

    return actual_decorator


def get_constants_maps(data: pd.DataFrame,
                       constant_names: Union[list, tuple, np.array]):
    if constant_names:
        maps = {c_name: data[c_name].iloc[0] for c_name in constant_names}
        return maps
    else:
        return {}


def get_categorical_maps(dummies: pd.DataFrame):
    uniqs = [np.unique(dummies[col]) for col in dummies.columns]
    keys = [len(u) for u in uniqs]
    maps = [dict(zip(np.arange(k), u)) for k, u in zip(keys, uniqs)]
    maps = dict(zip(dummies.columns, maps))
    return maps


def split(data: pd.DataFrame, split_sizes: tuple, mode: str = "idxs", stratify: tuple = None):
    assert np.sum(split_sizes) == 1
    s1, s2 = split_sizes
    if mode == "idxs":
        index_arr = np.arange(data.shape[0])
        if stratify:
            train_index, val_index = train_test_split(index_arr, test_size=s2, random_state=42,
                                                      stratify=data[stratify])
        else:
            train_index, val_index = train_test_split(index_arr, test_size=s2, random_state=42)
        return train_index, val_index
    elif mode == "standard":
        if stratify:
            train_data, val_data = train_test_split(data, test_size=s2, random_state=42,
                                                    stratify=data[stratify])
        else:
            train_data, val_data = train_test_split(data, test_size=s2, random_state=42)
        return train_data, val_data
    else:
        raise NotImplementedError()


def check_constants(data: pd.DataFrame, splits: dict):
    constant_names = get_column_separation(data)["constant"]
    s = {"constant": constant_names}
    for key, value in splits.items():
        v = list(value)
        [v.remove(item) for item in constant_names if item in value]
        s[key] = v

    return s


def get_column_separation(data: pd.DataFrame,
                          fill_auto: bool = True,
                          immutable_splits: dict = None
                          ):
    separations = {"constant": [], "continuous": [], "categorical": [], "discrete": []}
    if immutable_splits is not None:
        immutable = np.concatenate(list(immutable_splits.values()))
        for key, value in zip(immutable_splits.keys(), immutable_splits.values()):
            separations[key] += value
    else:
        immutable = []

    for col_name in data.columns:
        if col_name in immutable:
            continue
        else:
            if fill_auto:
                t = get_col_type(data[col_name])
                separations[t].append(col_name)
            else:
                separations["continuous"].append(col_name)
    return separations


def categorize(dummies: pd.DataFrame, maps: dict):
    d = []
    for key in dummies.columns:
        col = dummies[key]
        reverse_map = {v: k for k, v in maps[key].items()}
        col = col.map(reverse_map)
        arr = to_categorical(col, num_classes=len(maps[key]))
        columns = [key + "_{}".format(idx) for idx in range(len(maps[key]))]
        d.append(pd.DataFrame(arr, columns=columns))
    return d, list(dummies.columns)


def get_col_type(column, categorical_threshold=10):
    # chatGPT
    """
    Определяет тип переменной.

    Параметры:
    column (array-like): набор значений переменной.
    categorical_threshold (int): порог уникальных значений, при котором переменная считается категориальной.

    Возвращает:
    str: тип переменной - "categorical", "discrete", "continuous" или "constant".
    """
    range_of_values = np.max(column) - np.min(column)
    if range_of_values == 0:
        return "constant"
    elif isinstance(column[0], str) or isinstance(column[0], bool):
        return "categorical"
    elif len(np.unique(column)) <= categorical_threshold:
        return "categorical"
    elif np.allclose(column, column.astype(int)):
        return "discrete"
    else:
        return "continuous"
