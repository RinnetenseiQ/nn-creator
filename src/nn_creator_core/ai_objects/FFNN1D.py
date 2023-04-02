import os
import pickle
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from joblib import dump, load
from nn_creator_core.ai_objects.builders.fnn1d import build_std_fnn1d_model
from nn_creator_core.models.custom_layers import OneHotDecoder
from nn_creator_core.models.new_ensemble import Ensemble
from nn_creator_core.preparators.preparer import Preparer
from nn_creator_core.preparators.preparer_wraper import PreparerWrapper
from nn_creator_core.utils.metric_index import calc_metrics
from nn_creator_core.utils.uncertainity import checker
from nn_creator_core.utils.utils import (split, get_column_separation,
                                         get_constants_maps, get_categorical_maps,
                                         categorize)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FFNN1D:
    def __init__(self,
                 preparer: Preparer = None,
                 scale="std",
                 ensemble_args: dict = None,
                 uncertainty_args: dict = None,
                 automl_args: dict = None,
                 **kwargs):

        self.automl_args = automl_args
        self.uncertainty_args = uncertainty_args
        self.ensemble_args = ensemble_args
        self.scale = scale

        self.preparer = preparer if preparer else PreparerWrapper()
        self.cfg_fitted = False
        self.cfg = {}

        if scale == "std":
            self.input_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scale == "MM":
            self.input_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise NotImplementedError()

    def get_training_progress(self):
        progress = []
        if self.use_ensemble:
            progress = self.model.get_progress()
        else:
            pass
        return progress

    def dump(self, path):
        os.makedirs(path, exist_ok=True)

        dump(self.input_scaler, '{}/input_scaler.joblib'.format(path))
        dump(self.target_scaler, '{}/target_scaler.joblib'.format(path))

        with open('{}/cfg.pkl'.format(path), 'wb') as f:
            self.cfg["model_type"] = "FNN1D"
            pickle.dump(self.cfg, f)

        # with open('{}/endecoder_train_history.pkl'.format(path), 'wb') as f:
        #     pickle.dump(self.endecoder_train_history, f)
        #
        # with open('{}/imdecoder_train_history.pkl'.format(path), 'wb') as f:
        #     pickle.dump(self.imdecoder_train_history, f)
        if self.ensemble_args:
            self.model.save(path)
        else:
            self.model.save("{}/model.h5".format(path))

    def load(self, path, **kwargs):

        defaultKwargs = {"load_mode": "economy"}
        kwargs = {**defaultKwargs, **kwargs}

        with open('{}/cfg.pkl'.format(path), 'rb') as fp:
            self.cfg = pickle.load(fp)

        # with open('{}/endecoder_train_history.pkl'.format(path), 'rb') as fp:
        #     self.endecoder_train_history = pickle.load(fp)
        #
        # with open('{}/imdecoder_train_history.pkl'.format(path), 'rb') as fp:
        #     self.imdecoder_train_history = pickle.load(fp)


        self.ensemble_args = self.cfg["ensemble_args"]
        self.automl_args = self.cfg["automl_args"]
        self.uncertainty_args = self.cfg["uncertainty_args"]

        self.input_scaler = load('{}/input_scaler.joblib'.format(path))
        self.target_scaler = load('{}/target_scaler.joblib'.format(path))

        if self.ensemble_args:
            self.model = Ensemble().load(path, load_mode=kwargs["load_mode"])
        else:
            self.model = load_model("{}/model.h5".format(path), custom_objects={'OneHotDecoder': OneHotDecoder})

        self.fitted = True
        self.cfg_fitted = True

        sample = pd.read_csv("{}/sample.txt".format(path), sep='\t').reset_index(drop=True)
        self.predict(sample)
        return self

    def fit(self,
            inputs,
            targets,
            data: pd.DataFrame,
            epochs=1,
            split_sizes=(0.7, 0.3),
            stratify=None,
            column_splits=None,
            formulas=None,
            verbose=1,
            **kwargs):

        data[:1].to_csv(kwargs["save_path"] + "/sample.txt", sep="\t", index=False)

        train_index, val_index = split(data, split_sizes, mode="idxs", stratify=stratify)
        self._fit_config(data,
                         inputs=inputs,
                         targets=targets,
                         split_idxs=[train_index, val_index],
                         column_splits=column_splits,
                         formulas=formulas,
                         **kwargs)

        Xs = [self._preprocess(d, mode="input") for d in [data.iloc[train_index],
                                                          data.iloc[val_index]]]
        Ys = [self._preprocess(d, mode="target") for d in [data.iloc[train_index],
                                                           data.iloc[val_index]]]

        adds1, input_data = [*zip(*Xs)]  # matrix transposing for list representation
        adds2, output_data = [*zip(*Ys)]

        train_X, val_X = input_data
        train_y, val_y = output_data

        if self.ensemble_args:
            # TODO: implement OneHotDecoder for Ensemble
            trainable_model = Ensemble(input_cfgs=self.cfg["input_cfg"],
                                       output_cfgs=self.cfg["output_cfg"],
                                       cfg=self.cfg,
                                       target_columns=self.ensemble_args["target_groups"],
                                       uncertainty_args=self.uncertainty_args,
                                       automl_args=self.automl_args
                                       )
            self.model = trainable_model  # remove later
        else:
            # TODO: add weighted loss to ensemble
            trainable_model, self.model = build_std_fnn1d_model(self.cfg["input_cfg"],
                                                                self.cfg["output_cfg"])
            n_linear = self.cfg["output_cfg"][0]
            n_dummies = len(self.cfg["output_cfg"][1])
            n_variables = n_linear + n_dummies
            linear_weight = n_linear / n_variables
            dummies_weigts = [1 / n_variables] * n_dummies
            self.cfg["output_weights"] = weights = [linear_weight, *dummies_weigts]
            losses = ["mean_squared_error"] + (["categorical_crossentropy"] * len(self.cfg["encoder_input_cfg"][1]))
            self.cfg["losses"] = losses
            trainable_model.compile(optimizer="adam", loss=losses, loss_weights=weights)

        es_callback = EarlyStopping(monitor="val_loss", patience=150, restore_best_weights=True)
        lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=150, min_delta=0)
        train_history = trainable_model.fit(train_X, train_y,
                                            epochs=epochs, shuffle=True,
                                            validation_data=(val_X, val_y),
                                            callbacks=[es_callback, lr_callback],
                                            verbose=verbose,
                                            save_path=kwargs["save_path"]
                                            )
        self.train_history = train_history
        self.fitted = True
        return train_history

    def predict(self, data: pd.DataFrame, *args, **kwargs):
        # TODO: refactor method to exclude "inputs" definition before preparer transformation,
        #  preprocess must returns inputs
        # l = np.concatenate(self.cfg["model_input_order"])
        # inputs = data[l]
        adds, X = self._preprocess(data, mode="input", *args, **kwargs)
        pred = self.model.predict(X)
        res = self._postprocess(data, pred, adds, *args, **kwargs)
        return res

    def evaluate(self, data: pd.DataFrame, metrics=None, is_split=False, *args, **kwargs):
        def_kwargs = {"targets": self.cfg["target_names"],
                      "ranges": [self.cfg["feature_minimas"], self.cfg["feature_maximas"]],
                      "split_idxs": self.cfg["split_idxs"]}
        kwargs = {**def_kwargs, **kwargs}
        pred = self.predict(data)
        metrics = metrics if metrics else []

        if is_split:
            results = [calc_metrics(data.iloc[idxs], pred.iloc[idxs], metrics, **kwargs) for idxs in
                       kwargs["split_idxs"]]
        else:
            results = calc_metrics(data, pred, metrics, **kwargs)

        return results

    def _fit_config(self,
                    data: pd.DataFrame,
                    inputs: list[str],
                    targets: list[str],
                    column_splits: dict,
                    split_idxs: Union[list, np.array, tuple],
                    *args,
                    **kwargs):
        if self.cfg_fitted:
            pass

        # self.cfg["use_ensemble"] = self.use_ensemble
        # self.cfg["use_automl"] = self.use_automl
        # self.cfg["check_uncertainty"] = True if self.check_uncertainty else False

        self.cfg["columns_separations"] = get_column_separation(data, immutable_splits=column_splits)
        self.cfg["constant_maps"] = get_constants_maps(data, self.cfg["columns_separations"]["constant"])
        self.cfg["input_names"] = inputs
        self.cfg["target_names"] = targets
        self.cfg["scale"] = self.scale
        self.cfg["uncertainty_args"] = self.uncertainty_args
        self.cfg["automl_args"] = self.automl_args
        self.cfg["ensemble_args"] = self.ensemble_args
        self.cfg["split_idxs"] = split_idxs

        self.cfg["feature_minimas"] = data.min()
        self.cfg["feature_maximas"] = data.max()

        self.cfg["original_order"] = data.columns.tolist()

        transformed_data = self.preparer.transform(data,
                                                   **kwargs) if self.preparer.fitted else self.preparer.fit_transform(
            data, **kwargs)

        train_index, _ = split_idxs
        linear_columns = self.cfg["columns_separations"]["discrete"] + self.cfg["columns_separations"]["continuous"]

        c1 = [name for name in transformed_data.columns if (name in inputs and name in linear_columns)]
        c2 = [name for name in transformed_data.columns if (name in targets and name in linear_columns)]

        input_linear_data = transformed_data[c1].iloc[train_index]
        target_linear_data = transformed_data[c2].iloc[train_index]

        self.input_scaler.fit(input_linear_data)
        self.target_scaler.fit(target_linear_data)

        d_names = [name for name in transformed_data.columns if name in self.cfg["columns_separations"]["categorical"]]
        self.cfg["categorical_maps"] = c_maps = get_categorical_maps(transformed_data[d_names])

        inp_d_cfg = [len(c_maps[key]) for key in c_maps.keys() if key in inputs]
        tar_d_cfg = [len(c_maps[key]) for key in c_maps.keys() if key in targets]

        input_cfg = [len(c1), inp_d_cfg]
        output_cfg = [len(c2), tar_d_cfg]

        self.cfg["input_cfg"] = input_cfg
        self.cfg["output_cfg"] = output_cfg

        input_dummies_order = [name for name in d_names if name in inputs]
        target_dummies_order = [name for name in d_names if name in targets]

        self.cfg["model_input_order"] = [c1, input_dummies_order]
        self.cfg["model_output_order"] = [c2, target_dummies_order]

        self.cfg_fitted = True

    def _preprocess(self, data: pd.DataFrame, mode: str = "input", *args, **kwargs):
        assert self.cfg_fitted
        transformed_data = self.preparer.transform(data, **kwargs)
        if mode == "input":
            c_names = [name for name in transformed_data if name in self.cfg["input_names"]]
            d = transformed_data[c_names]
            adds_columns = [item for item in transformed_data.columns
                            if item not in self.cfg["input_names"] + self.cfg["target_names"]]

            adds = transformed_data[adds_columns]
            scaler = self.input_scaler
            linear_order, dum_order = self.cfg["model_input_order"]

        elif mode == "target":
            c_names = [name for name in transformed_data if name in self.cfg["target_names"]]
            d = transformed_data[c_names]
            adds_columns = [item for item in transformed_data.columns
                            if item not in self.cfg["input_names"] + self.cfg["target_names"]]

            adds = transformed_data[adds_columns]
            scaler = self.target_scaler
            linear_order, dum_order = self.cfg["model_output_order"]

        else:
            raise NotImplementedError()

        separations = {"continuous": [], "categorical": [], "discrete": []}
        for c_name in d.columns:
            for key in separations.keys():
                if c_name in self.cfg["columns_separations"][key]:
                    separations[key].append(c_name)

        data_dict = {key: d[value] if value else pd.DataFrame() for key, value in
                     zip(separations.keys(), separations.values())}

        temp = []
        for key, value in self.cfg["constant_maps"].items():
            arr = np.repeat(value, data.shape[0]).reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[key])
            temp.append(df)

        constants = pd.concat(temp, axis=1).reset_index(drop=True) if temp else pd.DataFrame()
        a = self.cfg["input_names"] if mode == "input" else self.cfg["target_names"]
        s = [item for item in constants.columns if item in a]
        constants = constants[s]
        continuals = data_dict["continuous"]
        dummies = data_dict["categorical"]
        discrets = data_dict["discrete"]
        dummies, d_names = categorize(dummies[dum_order], self.cfg["categorical_maps"]) if dum_order else ([], [])

        linears = pd.concat([discrets.reset_index(drop=True),
                             continuals.reset_index(drop=True)], axis=1)

        linears = linears[linear_order]
        linears = pd.DataFrame(scaler.transform(linears), columns=linears.columns)

        adds = pd.concat([adds.reset_index(drop=True), constants.reset_index(drop=True)], axis=1)

        data4nn = [linears, *dummies]
        return adds, data4nn

    def _postprocess(self, initial_data: pd.DataFrame, predictions: np.array, adds: pd.DataFrame, *args, **kwargs):
        def linear_postproc(output, order, scaler, discrets):
            res = pd.DataFrame(scaler.inverse_transform(output), columns=order)
            for col in order:
                if col in discrets:
                    res[col] = res[col].round()
            return res

        def dum_postproc(output, order, maps):
            res = pd.DataFrame(output, columns=order)
            for col in order:
                res[col] = res[col].map(maps[col])
            return res

        # TODO: make possibility to transform part of data columns by same rules
        # inputs = self.preparer.transform(initial_data[np.concatenate(self.cfg["model_input_order"])], **kwargs)
        inputs = self.preparer.transform(initial_data, **kwargs)[np.concatenate(self.cfg["model_input_order"])]

        linear_order, dum_order = self.cfg["model_output_order"]

        if linear_order and dum_order:
            linear, dum = predictions
            linear = linear_postproc(linear, linear_order, self.target_scaler,
                                     self.cfg["columns_separations"]["discrete"]
                                     )
            dum = dum_postproc(dum, dum_order, self.cfg["categorical_maps"])

            to_concat = [adds.reset_index(drop=True),
                         inputs.reset_index(drop=True),
                         linear.reset_index(drop=True),
                         dum.reset_index(drop=True)]
        elif linear_order:
            linear = predictions
            linear = linear[0]
            linear = linear_postproc(linear, linear_order, self.target_scaler,
                                     self.cfg["columns_separations"]["discrete"]
                                     )
            to_concat = [adds.reset_index(drop=True),
                         inputs.reset_index(drop=True),
                         linear.reset_index(drop=True)]
        elif dum_order:
            dum = predictions
            dum = dum_postproc(dum, dum_order, self.cfg["categorical_maps"])
            to_concat = [adds.reset_index(drop=True),
                         inputs.reset_index(drop=True),
                         dum.reset_index(drop=True)]
        else:
            raise ValueError()

        temp = []
        for key, value in self.cfg["constant_maps"].items():
            arr = np.repeat(value, adds.shape[0]).reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[key])
            temp.append(df)

        constants = pd.concat(temp, axis=1).reset_index(drop=True)
        s = [item for item in constants.columns if item in self.cfg["target_names"]]
        constants = constants[s]
        to_concat.insert(1, constants)
        result = pd.concat(to_concat, axis=1)
        result = self.preparer.inverse_transform(result, **kwargs)
        return result


if __name__ == '__main__':
    data = pd.read_csv("data/off_design_join_prepared.txt", sep='\t').reset_index(drop=True)
    c = {"ignored": ["num"]}

    project_name = "AutoML_offdesign"
    path = project_name + "({})".format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    overall_path = "data/Saved/off_design/" + path
    os.makedirs(overall_path, exist_ok=True)

    input_names = list(data.columns[:18])
    target_names = [name for name in data.columns if name not in input_names]
    instance = FFNN1D(use_automl=True,
                      check_uncertainty=checker,
                      use_ensemble=True
                      )

    instance.fit(inputs=input_names,
                 targets=target_names,
                 data=data,
                 epochs=1000,
                 split_sizes=(0.7, 0.3),
                 stratify=None,
                 column_splits=c,
                 formulas=None,
                 verbose=1,
                 save_path=overall_path,
                 max_aml_trials=20,
                 additional_training_epochs=1000,
                 n_aml_attempts=3,
                 n_uncertainty_models=3)
    instance.dump(overall_path)
    new_instance = FFNN1D().load(overall_path)
    preds = new_instance.predict(data)
    print("")
