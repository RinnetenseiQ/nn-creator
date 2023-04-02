import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
from joblib import dump, load
from nn_creator_core.ai_objects.parents import Model1D
from nn_creator_core.ai_objects.builders.model_builder import ModelBuilderAE1D, assemble_main_model
from nn_creator_core.models.custom_layers import CUSTOM_OBJECTS
from nn_creator_core.models.uncwrapper import UncWrapper
from nn_creator_core.preparators.preparer import Preparer
from nn_creator_core.preparators.preparer_wraper import PreparerWrapper
from nn_creator_core.utils.metric_index import calc_metrics
from nn_creator_core.utils.utils import (default_kwargs, split, categorize,
                                         get_column_separation, get_constants_maps,
                                         get_categorical_maps)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.models import save_model, load_model, Model

import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional


def custom_save_model(model, filepath, overwrite=True, save_format=None, **kwargs):
    if type(model) in [UncWrapper]:
        model.dump(filepath)
    elif type(model) in [Model, Functional]:
        save_model(model, "{}.h5".format(filepath), overwrite, save_format, **kwargs)
    else:
        raise NotImplementedError()


def custom_load_model(filepath, custom_objects=None, compile=True,
                      safe_mode=True, is_ensemble=False, output_cfg=None, is_unc=False, **kwargs):
    if is_ensemble:
        raise NotImplementedError()
    elif is_unc:
        model = UncWrapper().load(filepath, cfg=output_cfg, custom_objects=custom_objects)
    else:
        model = load_model("{}.h5".format(filepath), custom_objects, compile, safe_mode, **kwargs)
    return model


class AE1D(Model1D):
    # class AE1D:
    def __init__(self,
                 encode_size: Union[str, float, int] = "adaptive",
                 preparer: Preparer = None,
                 scale: str = "std",
                 approach="ae",
                 ensemble_args: dict = None,
                 uncertainty_args: dict = None,
                 automl_args: dict = None,
                 **kwargs):

        assert approach in ["ae", "vae"]
        self.approach = approach
        self.encode_size = encode_size
        self.preparer = preparer if preparer else PreparerWrapper()
        self.cfg_fitted = False
        if scale == "std":
            self.input_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scale == "MM":
            self.input_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise NotImplementedError()

        self.automl_args = automl_args if automl_args else {}
        self.uncertainty_args = uncertainty_args if uncertainty_args else {}
        self.ensemble_args = ensemble_args if ensemble_args else {}
        self.cfg = {}

    @default_kwargs(es_patience=50)
    def fit(self,
            inputs: list[str],
            targets: list[str],
            data: pd.DataFrame,
            epochs: int = 1,
            split_sizes: Union[tuple, list, np.array] = (0.7, 0.3),
            stratify: list[str] = None,
            column_splits: dict = None,
            verbose: Union[str, int] = 1,
            initializer: str = 'glorot_uniform',
            *args, **kwargs):

        data[:1].to_csv(kwargs["save_path"] + "/sample.txt", sep="\t", index=False)

        train_index, val_index = split(data, split_sizes, stratify=stratify)
        self._fit_config(data,
                         inputs=inputs,
                         targets=targets,
                         split_idxs=[train_index, val_index],
                         column_splits=column_splits,
                         *args, **kwargs)

        data1 = [self._preprocess(d, mode="input", *args, **kwargs) for d in [data.iloc[train_index],
                                                                              data.iloc[val_index]]]
        data2 = [self._preprocess(d, mode="target", *args, **kwargs) for d in [data.iloc[train_index],
                                                                               data.iloc[val_index]]]

        adds1, data4nn1 = [*zip(*data1)]
        adds2, data4nn2 = [*zip(*data2)]

        train_X1, val_X1 = data4nn1
        train_X2, val_X2 = data4nn2

        self.initializer = initializer
        build_cfg = [self.cfg["encoder_input_cfg"], self.cfg["imitator_input_cfg"], self.encode_size, initializer]
        builder = ModelBuilderAE1D(model_type=self.approach,
                                   cfg=build_cfg,
                                   uncertainty_arg=self.uncertainty_args
                                   )
        parts, trainable, self.model = builder.build()

        self.encoder, self.decoder, self.imitator = parts
        self.endecoder, self.imdecoder = trainable

        n_linear = self.cfg["encoder_input_cfg"][0]
        n_dummies = len(self.cfg["encoder_input_cfg"][1])
        n_variables = n_linear + n_dummies
        linear_weight = n_linear / n_variables
        dummies_weigts = [1 / n_variables] * n_dummies
        weights = [linear_weight, *dummies_weigts]
        self.cfg["output_weights"] = weights

        losses = ["mean_squared_error"] + (["categorical_crossentropy"] * len(self.cfg["encoder_input_cfg"][1]))
        self.cfg["losses"] = losses

        self.endecoder.compile(optimizer="adam", loss=losses, loss_weights=weights)
        self.imdecoder.compile(optimizer="adam", loss=losses, loss_weights=weights)

        es_callback = EarlyStopping(monitor="val_loss", patience=kwargs["es_patience"], restore_best_weights=True)
        lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=150, min_delta=0)
        endecoder_train_history = self.endecoder.fit(train_X2, train_X2,
                                                     epochs=epochs, shuffle=True,
                                                     validation_data=(val_X2, val_X2),
                                                     callbacks=[es_callback, lr_callback],
                                                     verbose=verbose)
        self.endecoder_train_history = endecoder_train_history if self.uncertainty_args else endecoder_train_history.history

        self.decoder.trainable = False
        es_callback = EarlyStopping(monitor="val_loss", patience=kwargs["es_patience"], restore_best_weights=True)
        lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=150, min_delta=0)
        imdecoder_train_history = self.imdecoder.fit(train_X1, train_X2,
                                                     epochs=epochs, shuffle=True,
                                                     validation_data=(val_X1, val_X2),
                                                     callbacks=[es_callback, lr_callback],
                                                     verbose=verbose)
        self.imdecoder_train_history = imdecoder_train_history if self.uncertainty_args else imdecoder_train_history.history
        res = self.imdecoder.predict(train_X1)
        self.model = assemble_main_model(self.imdecoder.model,
                                         build_cfg) if self.uncertainty_args and not self.model else self.model

        return [self.endecoder_train_history, self.imdecoder_train_history]

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

    def predict(self, data: pd.DataFrame, *args, **kwargs):
        # TODO: refactor method to exclude "inputs" definition before preparer transformation,
        #  preprocess must returns inputs
        # l = np.concatenate(self.cfg["model_input_order"])
        # inputs = data[l]
        adds, X = self._preprocess(data, mode="input", *args, **kwargs)
        pred = self.model.predict(X)
        res = self._postprocess(data, pred, adds, *args, **kwargs)
        return res

    def check_uncertainty(self, data: pd.DataFrame, *args, **kwargs):
        adds, X = self._preprocess(data, mode="input", *args, **kwargs)
        pred = self.imdecoder.check_uncertainty(X)
        return [tf.math.reduce_mean(x, axis=0) for x in pred]

    def dump(self, path: str = "", *args, **kwargs):

        os.makedirs(path, exist_ok=True)
        dump(self.input_scaler, '{}/input_scaler.joblib'.format(path))
        dump(self.target_scaler, '{}/target_scaler.joblib'.format(path))

        self.cfg["approach"] = self.approach
        self.cfg['initializer'] = self.initializer

        with open('{}/cfg.pkl'.format(path), 'wb') as f:
            self.cfg["model_type"] = "AE1D"
            pickle.dump(self.cfg, f)

        with open('{}/endecoder_train_history.pkl'.format(path), 'wb') as f:
            pickle.dump(self.endecoder_train_history, f)

        with open('{}/imdecoder_train_history.pkl'.format(path), 'wb') as f:
            pickle.dump(self.imdecoder_train_history, f)

        custom_save_model(self.encoder, "{}/encoder".format(path))
        custom_save_model(self.decoder, "{}/decoder".format(path))
        custom_save_model(self.imitator, "{}/imitator".format(path))

        custom_save_model(self.endecoder, "{}/endecoder".format(path))
        custom_save_model(self.imdecoder, "{}/imdecoder".format(path))
        custom_save_model(self.model, "{}/model".format(path))

        self.preparer.dump("{}/preparer".format(path))

    def load(self, path: str = "", *args, **kwargs):
        with open('{}/cfg.pkl'.format(path), 'rb') as fp:
            self.cfg = pickle.load(fp)

        # with open('{}/endecoder_train_history.pkl'.format(path), 'rb') as fp:
        #     self.endecoder_train_history = pickle.load(fp)
        #
        # with open('{}/imdecoder_train_history.pkl'.format(path), 'rb') as fp:
        #     self.imdecoder_train_history = pickle.load(fp)

        self.input_scaler = load('{}/input_scaler.joblib'.format(path))
        self.target_scaler = load('{}/target_scaler.joblib'.format(path))

        is_unc = bool(self.cfg["uncertainty_args"])
        is_ensemble = bool(self.cfg["ensemble_args"])
        self.model = custom_load_model("{}/model".format(path), custom_objects=CUSTOM_OBJECTS)
        # self.encoder = custom_load_model("{}/encoder".format(path), output_cfg=[self.cfg["ecode_size"], []], custom_objects=CUSTOM_OBJECTS)
        self.decoder = custom_load_model("{}/decoder".format(path), is_ensemble=is_ensemble, is_unc=is_unc,
                                         output_cfg=self.cfg["encoder_input_cfg"], custom_objects=CUSTOM_OBJECTS)
        # self.imitator = custom_load_model("{}/imitator".format(path), custom_objects=CUSTOM_OBJECTS)
        self.endecoder = custom_load_model("{}/endecoder".format(path), is_ensemble=is_ensemble, is_unc=is_unc,
                                           output_cfg=self.cfg["encoder_input_cfg"], custom_objects=CUSTOM_OBJECTS)
        self.imdecoder = custom_load_model("{}/imdecoder".format(path), is_ensemble=is_ensemble, is_unc=is_unc,
                                           output_cfg=self.cfg["encoder_input_cfg"], custom_objects=CUSTOM_OBJECTS)

        self.preparer = PreparerWrapper().load("{}/preparer".format(path))

        self.cfg_fitted = True
        return self

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

        if self.encode_size == "adaptive":
            self.encode_size = round(len(targets) / 2)
        elif type(self.encode_size) == float and 0 < self.encode_size <= 1:
            self.encode_size = round(len(targets) * self.encode_size)
        elif type(self.encode_size) == int:
            self.encode_size = self.encode_size
        else:
            raise ValueError()

        self.cfg["automl_args"] = self.automl_args
        self.cfg["uncertainty_args"] = self.uncertainty_args
        self.cfg["ensemble_args"] = self.ensemble_args

        self.cfg["input_names"] = inputs
        self.cfg["target_names"] = targets
        self.cfg["feature_minimas"] = data.min()
        self.cfg["feature_maximas"] = data.max()
        self.cfg["split_idxs"] = split_idxs

        self.cfg["columns_separations"] = get_column_separation(data, immutable_splits=column_splits)
        self.cfg["constant_maps"] = get_constants_maps(data, self.cfg["columns_separations"]["constant"])

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

        encoder_cfg = [len(c2), tar_d_cfg]
        imitator_cfg = [len(c1), inp_d_cfg]

        self.cfg["encoder_input_cfg"] = encoder_cfg
        self.cfg["imitator_input_cfg"] = imitator_cfg

        input_dummies_order = [name for name in d_names if name in inputs]
        target_dummies_order = [name for name in d_names if name in targets]

        self.cfg["model_input_order"] = [c1, input_dummies_order]
        self.cfg["model_output_order"] = [c2, target_dummies_order]

        self.cfg_fitted = True


if __name__ == "__main__":
    pass
