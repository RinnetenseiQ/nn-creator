from abc import ABC, abstractmethod

import pandas as pd


class Preparer(ABC):

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def dump(self, path: str, **kwargs):
        pass

    @abstractmethod
    def load(self, path: str, **kwargs):
        pass


