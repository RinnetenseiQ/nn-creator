import numpy as np
import pandas as pd
from solver.master.preparators.preparer import Preparer


def convert_to_deltas(data: pd.DataFrame):
    df = data.copy()
    df["delta_MFR"] = df["MFR_choke"] - df["MFR_surge"]
    df["delta_psr"] = df["psr_a_surge"] - df["psr_a_choke"]
    df["delta_ptr"] = df["ptr_a_surge"] - df["ptr_a_choke"]
    df = df.drop(["MFR_surge", "psr_a_choke", "ptr_a_choke"], axis=1)
    return df


def convert_from_deltas(data):
    df = data.copy()
    df["MFR_surge"] = df["MFR_choke"] - df["delta_MFR"]
    df["psr_a_choke"] = df["psr_a_surge"] - df["delta_psr"]
    df["ptr_a_choke"] = df["ptr_a_surge"] - df["delta_ptr"]
    df = df.drop(["delta_MFR", "delta_psr", "delta_ptr"], axis=1)
    return df


class DeltaPreparer(Preparer):
    def __init__(self, with_log=False, drop_subtractor=True):
        self.drop_subtractor = drop_subtractor
        self.with_log = with_log
        self.column_order = None
        self.fitted = False

    def fit(self, data: pd.DataFrame, **kwargs):
        self.column_order = list(data.columns)
        self.fitted = True

    def fit_transform(self, data: pd.DataFrame, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def transform(self, data: pd.DataFrame, **kwargs):
        assert self.fitted
        df = data.copy()
        df["delta_MFR"] = np.log(df["MFR_choke"] - df["MFR_surge"]) if self.with_log else df["MFR_choke"] - df["MFR_surge"]
        df["delta_psr"] = np.log(df["psr_a_surge"] - df["psr_a_choke"]) if self.with_log else df["psr_a_surge"] - df["psr_a_choke"]
        df["delta_ptr"] = np.log(df["ptr_a_surge"] - df["ptr_a_choke"]) if self.with_log else df["ptr_a_surge"] - df["ptr_a_choke"]
        df = df.drop(["MFR_surge", "psr_a_choke", "ptr_a_choke"], axis=1) if self.drop_subtractor else df
        return df

        # return convert_to_deltas(data)

    def inverse_transform(self, data: pd.DataFrame, **kwargs):
        df = data.copy()
        df["MFR_surge"] = df["MFR_choke"] - np.exp(df["delta_MFR"]) if self.with_log else df["MFR_choke"] - df["delta_MFR"]
        df["psr_a_choke"] = df["psr_a_surge"] - np.exp(df["delta_psr"]) if self.with_log else df["psr_a_surge"] - df["delta_psr"]
        df["ptr_a_choke"] = df["ptr_a_surge"] - np.exp(df["delta_ptr"]) if self.with_log else df["ptr_a_surge"] - df["delta_ptr"]
        df = df.drop(["delta_MFR", "delta_psr", "delta_ptr"], axis=1) if self.drop_subtractor else df
        return df
        # return convert_from_deltas(data)

    def dump(self, path, **kwargs):
        pass

    def load(self, path, **kwargs):
        pass


if __name__ == '__main__':
    data = pd.read_csv("data/off_design_join_prepared4_2.txt", sep='\t').reset_index(drop=True)
    instance = DeltaPreparer(with_log=True, drop_subtractor=True)

    prepared = instance.fit_transform(data)
    restored = instance.inverse_transform(prepared)

    a = (data - restored).sum(axis=0)
    print("")






