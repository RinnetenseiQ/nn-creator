import pandas as pd
import numpy as np


c1 = ["z.1", "b.2", "t.1", "a.1", "K1_d.1", "K2_d.1", "w1.2", "w2.2", "r1.2", "r2.2"]
c2 = ["b.1", "w1.1", "w2.1", "r1.1", "r2.1", "angOffTE.1"]
c3 = ["D1t.1", "D1h.1"]
c4 = ["D2t.2", "D2h.2"]
c5 = ["lc1.1", "lc2.2"]


def zero_condition(value, condition_value):
    return 0 if condition_value == 0 else value


def kVLD_condition(kVLD, layout):
    return 1.08 if layout != 0 else kVLD


def condition1(D2h, kVLD, layout):
    return 0 if layout == 0 else D2h * kVLD


def condition2(D2h, kVLD, kVD, layout):
    return 0 if layout == 0 else D2h * kVLD * kVD


def condition3(drr_rel, lc2, layout):
    return 0 if layout == 0 else lc2 + drr_rel * lc2


def formula2function(formula):
    pass


def zero_func(df: pd.DataFrame):
    pass


def k1d_formula(df: pd.DataFrame):
    t = df.copy()
    t["K1_d"] = b = np.arctan(t["CzU"]) * 180 / np.pi + t["ia"]
    return t


def kvd_formula(df):
    t = df.copy()
    t["k_VD"] = t.apply(lambda x: 0 if x["layout"] == 0 else x["k_VD"], axis=1)
    return t


def kvld_formula(df):
    t = df.copy()
    t["k_VLD"] = t.apply(lambda x: 1.08 if x["layout"] == 1 else x["k_VLD"], axis=1)
    return t


def zero_condition_functor(value_key, condition_key):
    def f(df: pd.DataFrame):
        t = df.copy()
        t[value_key] = t.apply(lambda x: zero_condition(x[value_key], x[condition_key]), axis=1)
        return t

    return f


ZERO_LAYOUT_FUNCTIONS = {key: zero_condition_functor(key, "layout") for key in
                         ["z.1", "b.2", "t.1", "a.1", "K1_d.1", "K2_d.1", "w1.2", "w2.2", "r1.2", "r2.2"]}
ZERO_SPLT_FUNCTIONS = {key: zero_condition_functor(key, "splt") for key in
                       ["b.1", "w1.1", "w2.1", "r1.1", "r2.1", "angOffTE.1"]}


def lc_formula_functor(key):
    def f(df: pd.DataFrame):
        t = df.copy()
        t[key] = t.apply(lambda x: condition3(x["drr_rel"], x["lc2"], x["layout"]), axis=1)
        return t

    return f


LC_FUNCTIONS = {key: lc_formula_functor(key) for key in ["lc1.1", "lc2.2"]}


def d1_formula_functor(key):
    def f(df: pd.DataFrame):
        t = df.copy()
        t[key] = t.apply(lambda x: condition1(x["D2h"], x["k_VLD"], x["layout"]), axis=1)
        return t

    return f


D1_FUNCTIONS = {key: d1_formula_functor(key) for key in ["D1t.1", "D1h.1"]}


def d2_formula_functor(key):
    def f(df: pd.DataFrame):
        t = df.copy()
        t[key] = t.apply(lambda x: condition2(x["D2h"], x["k_VLD"], x["k_VD"], x["layout"]), axis=1)
        return t

    return f


D2_FUNCTIONS = {key: d2_formula_functor(key) for key in ["D2t.2", "D2h.2"]}


def post_proc(df: pd.DataFrame):
    t = df.copy()
    for key in ["z.1", "b.2", "t.1", "a.1", "K1_d.1", "K2_d.1", "w1.2", "w2.2", "r1.2", "r2.2"]:
        t[key] = t.apply(lambda x: zero_condition(x[key], x["layout"]), axis=1)

    for key in ["b.1", "w1.1", "w2.1", "r1.1", "r2.1", "angOffTE.1"]:
        t[key] = t.apply(lambda x: zero_condition(x[key], x["splt"]), axis=1)

    t["k_VLD"] = t.apply(lambda x: kVLD_condition(x["k_VLD"], x["layout"]), axis=1)

    for key in ["D1t.1", "D1h.1"]:
        t[key] = t.apply(lambda x: condition1(x["D2h"], x["k_VLD"], x["layout"]), axis=1)

    for key in ["D2t.2", "D2h.2"]:
        t[key] = t.apply(lambda x: condition2(x["D2h"], x["k_VLD"], x["k_VD"], x["layout"]), axis=1)

    for key in ["lc1.1", "lc2.2"]:
        t[key] = t.apply(lambda x: condition3(x["drr_rel"], x["lc2"], x["layout"]), axis=1)

    t[["w1", "w2"]] = 2
    t["dar"] = 1.5

    return t