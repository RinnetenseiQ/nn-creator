import functools
import os

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import cloudpickle
from solver.master.preparators.preparer import Preparer
from solver.master.preparators.sources.formulas import kvd_formula, ZERO_LAYOUT_FUNCTIONS, kvld_formula, \
    ZERO_SPLT_FUNCTIONS, LC_FUNCTIONS, D1_FUNCTIONS, D2_FUNCTIONS, post_proc, k1d_formula
from solver.master.utils.utils import default_kwargs


def apply_formulas(data, formulas):
    literal = "arxhdkl_wqmasmdvcalsnd"
    columns = [name.replace(".", literal) for name in data.columns]
    data.columns = columns

    for formula in formulas:
        f = formula.replace(".", literal)
        data.eval(f, inplace=True)

    columns = [name.replace(literal, ".") for name in data.columns]
    data.columns = columns
    return data


def parse_feature_names(formulas):
    f = [formula.split("=")[0].replace(" ", "") for formula in formulas]
    return f


class FormulaPreparer(Preparer):
    def __init__(self,
                 formulas: list = None,
                 functions: dict = None,
                 input_names: list = None,
                 target_names: list = None,
                 overwrite: bool = True):
        self.target_names = target_names
        self.input_names = input_names
        self.functions = functions
        self.formulas = formulas
        self.overwrite = overwrite
        self.features_ranges = None
        self.fitted = False

    @default_kwargs(verify_transformation=True)
    def fit(self, data: pd.DataFrame, **kwargs):
        self.features_ranges = {"mins": data.min(axis=0).to_dict(),
                                "mean": data.mean(axis=0).to_dict(),
                                "maxs": data.max(axis=0).to_dict()}
        # TODO: introduce correlation evaluation
        self.calculatable_vars = parse_feature_names(self.formulas) if self.formulas else None
        self.formulas_dict = dict(zip(self.calculatable_vars, self.formulas))
        self.calculatable_vars = self.calculatable_vars + list(
            self.functions.keys()) if self.functions else self.calculatable_vars
        self.to_apply_before = [item for item in self.calculatable_vars if item in self.input_names]
        self.to_apply_after = [item for item in self.calculatable_vars if item in self.target_names]
        # TODO: remove to_drop
        self.to_drop = kwargs["to_drop"]
        # self.to_drop = [item for item in self.calculatable_vars if item in self.target_names]
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
    def fit_transform(self, data, **kwargs):
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def transform(self, data: pd.DataFrame, **kwargs):
        assert self.fitted
        if self.calculatable_vars:
            columns = list(data.columns)
            to_drop = [item for item in self.to_drop if item in columns]
            df = data.copy().drop(to_drop, axis=1)
            formulas = [self.formulas_dict[item] for item in self.to_apply_before if item in self.formulas_dict.keys()]
            functions = [self.functions[item] for item in self.to_apply_before if item in self.functions.keys()]

            for f in functions: df = f(df)
            df = apply_formulas(df, formulas) if formulas else df
            return df
        else:
            return data

    def inverse_transform(self, data: pd.DataFrame, **kwargs):
        assert self.fitted
        if self.calculatable_vars:
            df = data.copy()
            formulas = [self.formulas_dict[item] for item in self.to_apply_after if item in self.formulas_dict.keys()]
            functions = [self.functions[item] for item in self.to_apply_after if item in self.functions.keys()]

            df = apply_formulas(df, formulas) if formulas else df
            for f in functions: df = f(df)
            return df

        else:
            return data

    def dump(self, path, **kwargs):
        assert self.fitted
        os.makedirs(path, exist_ok=True)
        cfg = {}
        cfg["features_ranges"] = self.features_ranges
        cfg["calculatable_vars"] = self.calculatable_vars
        cfg["formulas_dict"] = self.formulas_dict
        cfg["functions_keys"] = list(self.functions.keys())
        cfg["to_drop"] = self.to_drop

        cfg["to_apply_before"] = self.to_apply_before
        cfg["to_apply_after"] = self.to_apply_after
        cfg["target_names"] = self.target_names
        cfg["input_names"] = self.input_names
        cfg["overwrite"] = self.overwrite

        with open("{}/cfg.pkl".format(path), "wb") as f:
            pickle.dump(cfg, f)

        for key in self.functions.keys():
            with open("{}/{}.pkl".format(path, key), "wb") as f:
                pickler = cloudpickle.CloudPickler(f)
                pickler.dump(self.functions[key])


    def load(self, path, **kwargs):
        with open("{}/cfg.pkl".format(path), "rb") as f:
            cfg = pickle.load(f)

        self.calculatable_vars = cfg["calculatable_vars"]
        self.formulas_dict = cfg["formulas_dict"]
        self.to_drop = cfg["to_drop"]
        self.features_ranges = cfg["features_ranges"]

        self.to_apply_before = cfg["to_apply_before"]
        self.to_apply_after = cfg["to_apply_after"]
        self.target_names = cfg["target_names"]
        self.input_names = cfg["input_names"]
        self.overwrite = cfg["overwrite"]

        self.functions = {}
        for key in cfg["functions_keys"]:
            with open("{}/{}.pkl".format(path, key), "rb") as f:
                self.functions[key] = pickle.load(f)

        self.fitted = True
        return self


if __name__ == '__main__':
    compressors_data = pd.read_csv('data/datasets/geometry/geom.txt', sep='\t').reset_index(drop=True)

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
    TO_DROP = ["drr", "lc1", "lc2.1", "lc1.2", "D2t", "w1", "w2", "dar", "D1t", "D2t.1", "D2h.1", "K1_d", "lc1.1",
               "lc2.2", "D1t.1", "D1h.1", "D2t.2", "D2h.2"]

    a = list(FUNCTIONS.keys())
    first_way_data = compressors_data.copy()
    first_way_data["k_VD"] = first_way_data.apply(lambda x: 0 if x["layout"] == 0 else x["k_VD"], axis=1)
    first_way_data["k_VLD"] = first_way_data.apply(lambda x: 1.08 if x["layout"] == 1 else x["k_VLD"], axis=1)

    restored1 = post_proc(first_way_data)
    diff = (first_way_data - restored1).sum(axis=0)

    instance = FormulaPreparer(formulas=FORMULAS, functions=FUNCTIONS,
                               input_names=inputs_names, target_names=target_names)

    compressors_data["k_VD"] = compressors_data.apply(lambda x: 0 if x["layout"] == 0 else x["k_VD"], axis=1)
    compressors_data["k_VLD"] = compressors_data.apply(lambda x: 1.08 if x["layout"] == 1 else x["k_VLD"], axis=1)

    transformed = instance.fit_transform(compressors_data, to_drop=TO_DROP)
    instance.dump("data/Saved/preparers/test_formula_preparer")
    instance = FormulaPreparer().load("data/Saved/preparers/test_formula_preparer")
    restored = instance.inverse_transform(transformed)

    res1 = (restored - transformed).sum(axis=0)
    res2 = (compressors_data - restored).sum(axis=0)
    res3 = (compressors_data - transformed).sum(axis=0)


    print("")
