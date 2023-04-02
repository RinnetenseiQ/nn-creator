import functools
import pandas as pd
import numpy as np
import os
import pickle

from solver.master.preparators.formulas_preparer import FormulaPreparer
from solver.master.preparators.preparer import Preparer
from solver.master.preparators.sources.formulas import kvd_formula, kvld_formula, ZERO_LAYOUT_FUNCTIONS, \
    ZERO_SPLT_FUNCTIONS, LC_FUNCTIONS, D1_FUNCTIONS, D2_FUNCTIONS
from solver.master.utils.utils import default_kwargs


class PreparerWrapper(Preparer):
    def __init__(self, preparers: list[Preparer] = None, order=None):
        if preparers and order:
            self.order = np.array(order)
            prep = np.array(preparers)
            inds = self.order.argsort()
            self.preparers = prep[inds]
        elif preparers and not order:
            self.order = np.arange(len(preparers))
            self.preparers = np.array(preparers)
        elif not preparers and order:
            raise ValueError()
        else:
            self.preparers = []
            self.order = []
        self.fitted = np.all([preparer.fitted for preparer in self.preparers])

    @default_kwargs(verify_transformation=True)
    def fit(self, data: pd.DataFrame, **kwargs):
        df = data.copy()
        for preparer in self.preparers:
            df = preparer.fit_transform(df, **kwargs)

        self.fitted = True

        if kwargs["verify_transformation"]:
            transformed = self.transform(data)
            restored = self.inverse_transform(transformed)
            diff = (data - restored).sum(axis=0)

            flag = np.isclose(diff.values, 0, atol=1e-03, )
            res = diff.to_frame(name="deviation")
            res["is_ok"] = flag
            self.data_verification = res

    @default_kwargs(verify_transformation=True)
    def fit_transform(self, data: pd.DataFrame, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def transform(self, data: pd.DataFrame, **kwargs):
        assert self.fitted
        df = data.copy()
        for preparer in self.preparers:
            df = preparer.transform(df, **kwargs)

        return df

    def inverse_transform(self, data: pd.DataFrame, **kwargs):
        assert self.fitted
        df = data.copy()

        for preparer in np.flip(self.preparers):
            df = preparer.inverse_transform(df)
        return df

    def dump(self, path, **kwargs):
        assert self.fitted
        os.makedirs(path, exist_ok=True)
        cfg = {}
        cfg["preparers_names"] = [preparer.__class__.__name__ for preparer in self.preparers] if self.preparers else []
        cfg["order"] = self.order
        with open("{}/cfg.pkl".format(path), "wb") as f:
            pickle.dump(cfg, f)

        for preparer in self.preparers:
            preparer_path = "{}/{}".format(path, preparer.__class__.__name__)
            os.makedirs(preparer_path, exist_ok=True)
            preparer.dump(preparer_path)

    def load(self, path, **kwargs):
        with open("{}/cfg.pkl".format(path), "rb") as f:
            cfg = pickle.load(f)

        self.order = cfg["order"]
        self.preparers = [eval(preparer_name)().load("{}/{}".format(path, preparer_name))
                          for preparer_name in cfg["preparers_names"]]
        self.fitted = True
        return self


if __name__ == '__main__':
    compressors_data = pd.read_csv('data/datasets/geometry/geom.txt', sep='\t').reset_index(drop=True)
    compressors_data["k_VD"] = compressors_data.apply(lambda x: 0 if x["layout"] == 0 else x["k_VD"], axis=1)
    compressors_data["k_VLD"] = compressors_data.apply(lambda x: 1.08 if x["layout"] == 1 else x["k_VLD"], axis=1)

    inputs_names = ['layout', 'ptr', 'mfr', 'ns', 'LuUU', 'CzU', 'dbt',
                    'Czm_Cz1', 'Zrr', 'BL', 'k_VLD',
                    'k_VD', 'B', 'ia', 'splt', "drr_rel"]
    target_names = [name for name in compressors_data.columns if name not in inputs_names + ["index"]]
    FORMULAS = ["drr = drr_rel * lc2",
                "lc1 = lc2 + drr_rel * lc2",
                "lc2.1 = lc2 + drr_rel * lc2",
                "lc1.2 = lc2 + drr_rel * lc2",
                "D2t = D2h",
                "w1 = 2",
                "w2 = 2",
                "dar = 3/2",
                "D1t = D1h / dbt",
                "D2t.1 = D2h * k_VLD",
                "D2h.1 = D2h * k_VLD",
                ]
    FUNCTIONS = {  # "K1_d": k1d_formula,
        "k_VD": kvd_formula,
        "k_VLD": kvld_formula,
        **ZERO_LAYOUT_FUNCTIONS,
        **ZERO_SPLT_FUNCTIONS,
        **LC_FUNCTIONS,
        **D1_FUNCTIONS,
        **D2_FUNCTIONS
    }
    TO_DROP = ["drr", "lc1", "lc2.1", "lc1.2", "D2t", "w1", "w2", "dar", "D1t", "D2t.1", "D2h.1", "lc1.1",
               "lc2.2", "D1t.1", "D1h.1", "D2t.2", "D2h.2"]

    formula_preparer = FormulaPreparer(formulas=FORMULAS, functions=FUNCTIONS,
                                       input_names=inputs_names, target_names=target_names)

    instance = PreparerWrapper(preparers=[formula_preparer])

    transformed = instance.fit_transform(compressors_data, to_drop=TO_DROP)
    instance.dump("data/Saved/preparers/test_wrapper_preparer")
    instance = PreparerWrapper().load("data/Saved/preparers/test_wrapper_preparer")
    restored = instance.inverse_transform(transformed)

    res1 = (restored - transformed).sum(axis=0)
    res2 = (compressors_data - restored).sum(axis=0)
    res3 = (compressors_data - transformed).sum(axis=0)

    print("")
