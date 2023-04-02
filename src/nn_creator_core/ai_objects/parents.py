from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import numpy as np


class Model1D(ABC):

    @abstractmethod
    def fit(self,
            inputs: list[str],
            targets: list[str],
            data: pd.DataFrame,
            epochs: int = 1,
            split_sizes: Union[tuple, list, np.array] = (0.7, 0.3),
            stratify: list[str] = None,
            column_splits: dict = None,
            verbose: Union[str, int] = 1,
            *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def dump(self, path: str, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, path: str, *args, **kwargs):
        pass

    @abstractmethod
    def _preprocess(self, data: pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def _postprocess(self, inputs: pd.DataFrame, predictions: np.array, adds: pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def _fit_config(self,
                    data: pd.DataFrame,
                    inputs: list[str],
                    targets: list[str],
                    column_splits: dict,
                    split_idxs: Union[tuple, list, np.array],
                    *args, **kwargs):
        pass
