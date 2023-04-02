import datetime
import os
import pickle
from typing import Union, Callable

import autokeras as ak
from joblib import dump

from tqdm import tqdm
from tqdm.contrib import tzip
import tensorflow as tf

import shutil
from datetime import datetime
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import average, concatenate
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# log_file_path = r"../../../../data/Saved/log.txt"


# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

INITIALIZERS = {"he_uniform": tf.keras.initializers.he_uniform}


def log_message(message, log_file_location, identation_level=0, t=None):
    with open("{}/log.txt".format(log_file_location), "a") as f:
        ident = "\t" * identation_level
        if t:
            f.write("##########################\n\n" +
                    "{}{} at {}".format(ident, message, t.strftime("%d-%m-%Y %H-%M")) + "\n")
        else:
            f.write("##########################\n\n" +
                    "{}{}".format(ident, message) + "\n")


def split_array(arr):
    return [arr[:, i].reshape(-1, 1) for i in range(arr.shape[-1])]


def custom_clone_model(base_model, initializer="he_uniform"):
    model_copy = tf.keras.models.clone_model(base_model)
    for layer in model_copy.layers:
        if layer is tf.keras.layers.Dense:
            layer.kernel_initializer = INITIALIZERS[initializer]()
    res = tf.keras.models.clone_model(model_copy)
    return res


def create_unc_module(base_path, unc_group_paths):
    models = [load_model(base_path + "/" + path) for path in unc_group_paths]
    for model, idx in zip(models, range(len(models))):
        model._name = "mod_{}".format(idx)
    main_inputs = models[0].inputs
    hiddens = [model(main_inputs) for model in models]
    # hidden_outputs = [item.outputs for item in models]
    output = average(hiddens)
    return Model(inputs=main_inputs, outputs=output), Model(inputs=main_inputs, outputs=hiddens)


def model_from_modules(modules):
    main_inputs = modules[0].inputs

    outputs = [module(main_inputs) for module in modules]

    return Model(inputs=main_inputs, outputs=outputs)


def assemble_models(base_path, paths, unc=True):
    if unc:
        average_modules = []
        pred_modules = []
        for model_path_group, idx in zip(paths, range(len(paths))):
            average_module, pred_module = create_unc_module(base_path, model_path_group)
            average_modules.append(average_module)
            pred_modules.append(pred_module)

        averaged_model = model_from_modules(average_modules)
        model = model_from_modules(average_modules)
        return averaged_model, model
    else:
        raise NotImplementedError()


def move_best_models(base_path, old_models_paths, n_uncertainty_models=0):
    temp_best_paths = []
    if n_uncertainty_models:
        suffix = [""] + ["_unc_{}".format(i) for i in range(n_uncertainty_models - 1)]

        for main_model_path in old_models_paths:
            models_paths = [p+s for p, s in zip(main_model_path, suffix)]
            # models_paths = [[p + s for p, s in zip(unc_paths, suffix)] for unc_paths in main_model_path]
            # for idx, model_path in enumerate(models_paths):
            for idx, model_path in enumerate(main_model_path):
                # target = p + "/best_models/" + main_model_path.split("/")[-1] + "/{}".format(idx)
                target = base_path + "/best_models/" + model_path.split("/")[-1]
                os.makedirs(target, exist_ok=True)
                shutil.move(base_path + "/" + model_path, target)
                old_name = target + "/" + model_path.split("/")[-1]
                new_name = "/".join(target.split("/") + ["{}".format(idx)])
                os.rename(old_name, new_name)
                path_to_save = "/".join(new_name.split("/")[-3:])
                temp_best_paths.append(path_to_save)
        best_models_paths = np.array(temp_best_paths).reshape(-1, n_uncertainty_models)
    else:
        raise NotImplementedError()
    return best_models_paths


class Ensemble:
    @default_kwargs(load_model="economy")
    def __init__(self,
                 input_cfgs=None,
                 output_cfgs=None,
                 target_columns=None,
                 # check_uncertainty: Union[bool, Callable] = False,
                 # use_automl=False,
                 cfg=None,
                 # n_uncertainty_models=3,
                 uncertainty_args: dict = None,
                 automl_args: dict = None,
                 *args, **kwargs):

        # super().__init__(*args, **kwargs)

        self.automl_args = automl_args
        self.uncertainty_args = uncertainty_args

        defaultKwargs = {"load_mode": "economy"}
        kwargs = {**defaultKwargs, **kwargs}

        self.target_columns = target_columns
        self.cfg = {} if cfg is None else cfg
        self.use_automl = bool(automl_args)
        self.output_cfgs = output_cfgs
        self.input_cfgs = input_cfgs

        self.check_uncertainty = True if check_uncertainty else False
        self.uncertainty_checker = check_uncertainty if callable(check_uncertainty) else default_checker
        self.n_uncertainty_models = uncertainty_args["n_models"] if self.uncertainty_args else 0

        self.best_models_paths = None
        col_type_separation = {}
        self.model_output_order = []

        if self.cfg:
            col_type_separation["linear"] = self.cfg["columns_separations"]["discrete"] + \
                                            self.cfg["columns_separations"]["continuous"]
            col_type_separation["categorical"] = self.cfg["columns_separations"]["categorical"]
            self.model_output_order = self.cfg["model_output_order"]
        self.col_type_separation = col_type_separation
        self.load_mode = kwargs["load_mode"]
        self.location = ""
        self.progress = {}

    def _define_progress(self, max_attempts=None):
        self.progress["current_attempt"] = 0
        self.progress["max_attempts"] = max_attempts
        self.progress["n_models_found"] = 0
        self.progress["total_targets"] = len(self.target_columns)
        self.progress["n_uncertainty_models"] = self.n_uncertainty_models
        self.progress["current_uncertainty_model"] = 0
        self.progress["current_task"] = "init"

    def _subset_output_data(self, data, target):
        train_y, val_y = data
        new_cfg = []
        sub_model_output = []
        temp_train_y = []
        temp_val_y = []

        if self.output_cfgs[0]:
            linear_cols = [item for item in target if item in self.model_output_order[0]]
            temp_train_y.append(train_y[0][linear_cols])
            temp_val_y.append(val_y[0][linear_cols])
            new_cfg.append(len(linear_cols))
            sub_model_output.append(linear_cols)
        else:
            new_cfg.append(0)
            sub_model_output.append([])

        if self.output_cfgs[1]:
            train_cat_dict = dict(zip(self.model_output_order[1], train_y[1:])) if self.output_cfgs[0] else dict(
                zip(self.model_output_order[1], train_y))
            val_cat_dict = dict(zip(self.model_output_order[1], val_y[1:])) if self.output_cfgs[0] else dict(
                zip(self.model_output_order[1], val_y))
            cfg_dict = dict(zip(self.model_output_order[1], self.output_cfgs[1]))
            cat_cols = [item for item in target if item in self.model_output_order[1]]
            train_categorical = [train_cat_dict[c_name] for c_name in cat_cols]
            val_categorical = [val_cat_dict[c_name] for c_name in cat_cols]
            cat_cfg = [cfg_dict[c_name] for c_name in cat_cols]
            new_cfg.append(cat_cfg)
            sub_model_output.append(cat_cols)
        else:
            train_categorical = []
            val_categorical = []
            new_cfg.append([])
            sub_model_output.append([])

        y_train = temp_train_y + train_categorical
        y_val = temp_val_y + val_categorical

        return y_train, y_val, new_cfg, sub_model_output

    def get_progress(self):
        return self.progress

    def predict(self,
                x,
                batch_size=None,
                verbose='auto',
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):

        if self.load_mode == "power":
            # full_preds = self.model.predict(x)
            predictions = self.model.predict(x)
            if self.linear_output_order and self.categorical_output_order:
                linear = pd.DataFrame(predictions[0], columns=self.linear_output_order)[self.model_output_order[0]].values
                cat_dict = dict(zip(self.categorical_output_order, predictions[1:]))
                categorical = [cat_dict[name] for name in self.model_output_order[1]]
                res = [linear, *categorical]
            elif self.linear_output_order:
                linear = pd.DataFrame(predictions[0], columns=self.linear_output_order)[self.model_output_order[0]].values
                res = [linear]
            elif self.categorical_output_order:
                cat_dict = dict(zip(self.categorical_output_order, predictions))
                categorical = [cat_dict[name] for name in self.model_output_order[1]]
                res = categorical
            else:
                raise ValueError()
            return res

        elif self.load_mode == "economy":
            predictions = []
            for model_path in self.best_models_paths:
                if self.check_uncertainty:
                    prediction, _ = self._predict_uncertainty(model_path, x, verbose=verbose)
                else:
                    prediction = self._predict(model_path, x, verbose=verbose)
                predictions.append(prediction)
        else:
            raise NotImplementedError()

    def _predict_uncertainty(self, models_paths, X, verbose="auto"):
        preds = []
        for path in models_paths:
            # TODO: bug with custom objects
            #
            model = load_model(self.location + "/" + path, custom_objects={"AdamWeightDecay": AdamW})
            preds.append(model.predict(X, verbose=verbose))

        average_predictions = np.array(preds).mean(axis=0)
        return average_predictions, preds

    def _predict(self, model_path, X, verbose="auto"):
        model = load_model(self.location + "/" + model_path, custom_objects={"AdamWeightDecay": AdamW})
        return model.predict(X, verbose=verbose)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose='auto',
            callbacks=None,
            validation_data=None,
            workers=1,
            use_multiprocessing=False,
            **kwargs):

        self._define_progress(self.automl_args["n_aml_attempts"])
        self.location = kwargs["save_path"]
        train_X = x
        train_y = y
        val_X, val_y = validation_data

        start_time = datetime.now()
        log_message("Training started", self.location, 0, start_time)

        best_models_path = []
        sub_cfgs = []
        sub_output_names = []
        p = kwargs["save_path"]
        if self.use_automl:
            os.makedirs(p + "/best_models")
            os.makedirs(p + "/temp")

        for target in tqdm(self.target_columns):
            sub_train_y, sub_val_y, sub_cfg, output_names = self._subset_output_data(data=[train_y, val_y],
                                                                                     target=target)
            sub_cfgs.append(sub_cfg)
            sub_output_names.append(output_names)
            if self.use_automl:
                model_path = self._find_architecture(data=[train_X, sub_train_y, val_X, sub_val_y],
                                                     output_cfg=sub_cfg,
                                                     target_name=target,
                                                     max_trials=self.automl_args["max_aml_trials"],
                                                     max_model_size=None,
                                                     epochs=epochs,
                                                     save_path=p)
            else:
                model_path = self._fit_manually()

            if self.check_uncertainty:
                model_path = self._train_uncertainty_models(data=[train_X, sub_train_y, val_X, sub_val_y],
                                                            output_cfg=sub_cfg,
                                                            target_name=target,
                                                            model_path=model_path,
                                                            n_models=self.n_uncertainty_models,
                                                            epochs=epochs)
            best_models_path.append(model_path)
        self.best_models_paths = move_best_models(base_path=p,
                                                  old_models_paths=best_models_path,
                                                  n_uncertainty_models=self.n_uncertainty_models)
        self.models_output_cfgs = sub_cfgs
        self.models_output_names = sub_output_names
        if self.load_mode == "power": self._assemble()

    def _assemble(self):
        models = []
        for path in self.best_models_paths:
            if self.check_uncertainty:
                models = [load_model("{}/{}".format(self.location, p)) for p in path]
                model = UncWrapper(models=models).model
            else:
                model = load_model("{}/{}".format(self.location, path))
            models.append(model)

        inputs = models[0].inputs
        linear_outputs = []
        linear_order = []

        cat_outputs = []
        categorical_order = []


        for target, cfg, order, model in zip(self.target_columns, self.models_output_cfgs,
                                             self.models_output_names, models):
            output = model(inputs)
            if cfg[0]:
                linear_outputs.append(output[0])
                linear_order += order[0]

            if cfg[1]:
                cat_outputs += output[1:] if cfg[0] else output
                categorical_order += order[1]


        linear_output = concatenate(linear_outputs)
        outputs = [linear_output, *cat_outputs]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.linear_output_order = linear_order
        self.categorical_output_order = categorical_order


    def _find_architecture(self, data,
                           output_cfg,
                           target_name,
                           max_trials=150,
                           max_model_size=None,
                           epochs=1000,
                           project_name='AutoAutoML',
                           save_path=""):
        train_X, train_y, val_X, val_y = data
        input_cfg = self.cfg["input_cfg"]

        tasks = ["regression"] if output_cfg[0] else []
        tasks = tasks + ["classification"] * (len(output_cfg[1]))

        inputs, outputs = build_ak_fnn1d_model(input_cfg, output_cfg, tasks=tasks)

        pn = str(target_name)
        s = os.path.normpath(save_path + "/temp/AutoKeras")

        model = ak.AutoModel(inputs=inputs, outputs=outputs,
                             max_trials=max_trials,
                             max_model_size=max_model_size,
                             overwrite=True,
                             project_name=pn,
                             directory=s)

        cb_list = [EarlyStopping(monitor="val_loss", patience=100,
                                 restore_best_weights=True),
                   TerminateOnNaN(),
                   # ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=50, min_delta=0)
                   ]

        model.fit(train_X, train_y,
                  validation_data=(val_X, val_y),
                  epochs=epochs, verbose=0,
                  callbacks=cb_list
                  )

        tf_model = model.export_model()
        path = "temp/" + project_name.split("/")[0] + f"/{target_name}"
        tf_model.save(self.location + "/" + path)

        return path

    def _fit_manually(self):
        raise NotImplementedError()

    def _train_uncertainty_models(self,
                                  data,
                                  output_cfg,
                                  target_name,
                                  model_path,
                                  n_models=6,
                                  epochs=5000):
        train_X, train_y, val_X, val_y = data
        main_model = tf.keras.models.load_model(self.location + "/" + model_path,
                                                custom_objects={"AdamWeightDecay": AdamW})

        models = [custom_clone_model(main_model, initializer="he_uniform") for _ in range(n_models - 1)]
        unc_model = UncWrapper(models=models, cfg=output_cfg, is_seq=True)

        losses = []
        if output_cfg[0]:
            losses.append("mean_squared_error")

        if output_cfg[1]:
            losses += ["categorical_crossentropy"] * len(output_cfg[1])

        unc_model.compile(optimizer="adam", loss=losses)
        unc_model.fit(X=train_X, y=train_y, epochs=epochs,
                      validation_data=(val_X, val_y),
                      callbacks=None,
                      shuffle=True,
                      verbose="auto")
        models = unc_model.models + [main_model]
        suffixes = [""] + ["_unc_{}".format(model_num) for model_num in range(n_models - 1)]
        unc_paths = []
        for model, s in zip(models, suffixes):
            path = "{}/{}{}".format(self.location, str(target_name), s)
            model.save(path)
            unc_paths.append(path)

        return unc_paths

    def load(self, filepath, load_mode="economy"):
        with open('{}/ensemble_cfg.pkl'.format(filepath), 'rb') as fp:
            dictionary = pickle.load(fp)

        self.target_columns = dictionary["target_columns"]
        self.use_automl = dictionary["use_automl"]
        self.output_cfgs = dictionary["output_cfgs"]
        self.input_cfgs = dictionary["input_cfgs"]
        self.check_uncertainty = dictionary["check_uncertainty"]
        self.n_uncertainty_models = dictionary["n_uncertainty_models"]
        self.col_type_separation = dictionary["col_type_separation"]
        self.best_models_paths = dictionary["best_models_paths"]
        self.model_output_order = dictionary["model_output_order"]
        self.location = filepath

        # self.uncertainty_checker = load(
        #     '{}/uncertainty_checker.joblib'.format(filepath)) if self.check_uncertainty else None
        self.load_mode = load_mode
        if self.load_mode == "power":
            # self.averaged_model, self.model = assemble_models(self.location, self.best_models_paths,
            #                                                   unc=self.check_uncertainty)
            self._assemble()

        elif self.load_mode == "economy":
            pass
        else:
            raise NotImplementedError()
        return self

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True
             ):
        dictionary = {"target_columns": self.target_columns, "use_automl": self.use_automl,
                      "output_cfgs": self.output_cfgs, "input_cfgs": self.input_cfgs,
                      "check_uncertainty": self.check_uncertainty, "n_uncertainty_models": self.n_uncertainty_models,
                      "col_type_separation": self.col_type_separation, "best_models_paths": self.best_models_paths,
                      "model_output_order": self.model_output_order}

        with open('{}/ensemble_cfg.pkl'.format(filepath), 'wb') as f:
            pickle.dump(dictionary, f)

        if self.check_uncertainty:
            dump(self.uncertainty_checker, '{}/uncertainty_checker.joblib'.format(filepath))
