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
from tensorflow.python.keras.layers import average
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# log_file_path = r"../../../../data/Saved/log.txt"


# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

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


class Ensemble:
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

    def get_progress(self):
        return self.progress

    # @Override
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
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("##########################\n\n" +
                    "Training started at {}".format(start_time.strftime("%d-%m-%Y %H-%M")) + "\n")

        if self.use_automl and self.check_uncertainty:
            p = kwargs["save_path"]
            os.makedirs(p + "/best_models")
            os.makedirs(p + "/temp")
            best_models_paths = self._get_models_with_good_enough_uncertainty(input_data=[train_X, val_X],
                                                                              output_data=[train_y, val_y],
                                                                              target_columns=self.target_columns,
                                                                              epochs=epochs,
                                                                              column_type_separation=self.col_type_separation,
                                                                              save_path=kwargs["save_path"],
                                                                              max_trials=self.automl_args["max_aml_trials"],
                                                                              additional_training_epochs=self.automl_args[
                                                                                 "additional_training_epochs"],
                                                                              n_attempts=self.automl_args["n_aml_attempts"],
                                                                              )

            suffix = [""] + ["_unc_{}".format(i) for i in range(self.n_uncertainty_models)]
            temp_best_paths = []
            for main_model_path in best_models_paths:
                models_paths = [main_model_path + s for s in suffix]
                for idx, model_path in enumerate(models_paths):
                    # target = p + "/best_models/" + main_model_path.split("/")[-1] + "/{}".format(idx)
                    target = p + "/best_models/" + main_model_path.split("/")[-1]
                    os.makedirs(target, exist_ok=True)
                    shutil.move(p + "/" + model_path, target)
                    old_name = target + "/" + model_path.split("/")[-1]
                    new_name = "/".join(target.split("/") + ["{}".format(idx)])
                    os.rename(old_name, new_name)
                    path_to_save = "/".join(new_name.split("/")[-3:])
                    temp_best_paths.append(path_to_save)
            self.best_models_paths = np.array(temp_best_paths).reshape(-1, self.n_uncertainty_models + 1)

        print("Done!!!")
        end_time = datetime.now()
        diff = end_time - start_time
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("Training ended at {}".format(end_time.strftime("%d-%m-%Y %H-%M")) + "\n")
            f.write("Training time: {}".format(diff) + "\n")
        return

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
            predictions = self.averaged_model.predict(x)
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

        flattened_preds = []
        output_order = []
        for p, target_name in zip(predictions, self.target_columns):
            # output_order.append(*target_name)
            output_order += target_name
            # flattened_preds.append(*p)
            flattened_preds += split_array(p)

        prediction_dict = dict(zip(output_order, flattened_preds))

        linear_order, categorical_order = self.model_output_order
        linear_preds = []
        for c_name in linear_order:
            linear_preds.append(prediction_dict[c_name])

        linear_preds = np.concatenate(linear_preds, axis=1)

        categorical_preds = []
        for c_name in categorical_order:
            categorical_preds.append(prediction_dict[c_name])

        result = [linear_preds, *categorical_preds]
        return result

    def call(self, inputs, training=None, mask=None):
        pass

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

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None,
             save_traces=True
             ):
        # TODO: make best_model_path related on filepath
        dictionary = {"target_columns": self.target_columns, "use_automl": self.use_automl,
                      "output_cfgs": self.output_cfgs, "input_cfgs": self.input_cfgs,
                      "check_uncertainty": self.check_uncertainty, "n_uncertainty_models": self.n_uncertainty_models,
                      "col_type_separation": self.col_type_separation, "best_models_paths": self.best_models_paths,
                      "model_output_order": self.model_output_order}

        with open('{}/ensemble_cfg.pkl'.format(filepath), 'wb') as f:
            pickle.dump(dictionary, f)

        if self.check_uncertainty:
            dump(self.uncertainty_checker, '{}/uncertainty_checker.joblib'.format(filepath))

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
            self.averaged_model, self.model = assemble_models(self.location, self.best_models_paths,
                                                              unc=self.check_uncertainty)
        elif self.load_mode == "economy":
            pass
        else:
            raise NotImplementedError()
        return self

    ####################################################
    #####################################################
    ###################################################

    def _train_uncertainty_models(self,
                                  input_data,
                                  output_data,
                                  target_columns,
                                  column_type_separation,
                                  model_paths,
                                  n_models=6,
                                  epochs=5000,
                                  load_path=""):
        # Trains additional n_models models with the same architecture
        self.progress["current_task"] = "uncertainty_training"
        start_time = datetime.now()
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Uncertainty training started".format(start_time.strftime("%d-%m-%Y %H-%M")) + "\n")

        train_X, val_X = input_data
        train_y, val_y = output_data
        # Prepare data

        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=50, min_delta=0)

        output_model_names = []

        for column_idx, c_name in tzip(range(len(target_columns)), target_columns, desc="Uncertainty training",
                                       leave=True):
            # Prepare target variables
            y_train, y_val = [get_partially_train_data(data, c_name, column_type_separation) for data in output_data]
            main_model = tf.keras.models.load_model(self.location + "/" + model_paths[column_idx],
                                                    custom_objects={"AdamWeightDecay": AdamW})
            # Find normalizarion layer (its parameters should be copied directly) and change layer initializers to he_uniform
            for layer in main_model.layers:
                if layer is tf.keras.layers.Dense:
                    layer.kernel_initializer = tf.keras.initializers.he_uniform()

            # print(f'\nTraining uncertainty models for {model_paths[column_idx]}')
            # Train extra models
            for model_num in range(n_models):
                model_copy = tf.keras.models.clone_model(main_model)
                model_copy.compile(loss='MSE', optimizer='adam')

                model_copy.fit(train_X, y_train, validation_data=(val_X, y_val),
                               epochs=epochs, callbacks=[es_callback, lr_callback], verbose=0)

                # path = load_path + "/" + f'{model_paths[column_idx]}_extra_{model_num}'
                # model_copy.save(f'{model_names[column_idx]}_extra_{model_num}')
                path = self.location + "/" + model_paths[column_idx] + "_unc_{}".format(model_num)
                model_copy.save(path)
            # print(f'Uncertainty models trained\n')
        end_time = datetime.now()
        diff = end_time - start_time
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Uncertainty training finished".format(end_time.strftime("%d-%m-%Y %H-%M")) + "\n")
            f.write("       Elapsed time: {}".format(diff) + "\n")

    def _train_some_more(self,
                         input_data,
                         output_data,
                         target_columns,
                         column_type_separation,
                         models_paths,
                         epochs=10000,
                         load_path=""):
        # Trains the models for additional epochs
        self.progress["current_task"] = "additional_training"
        start_time = datetime.now()
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Additional training started".format(start_time.strftime("%d-%m-%Y %H-%M")) + "\n")
        train_X, val_X = input_data
        train_y, val_y = output_data

        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=80, min_delta=0)

        output_model_names = []

        for column_idx, c_name in tzip(range(len(target_columns)), target_columns, desc="Additional training",
                                       leave=True):
            # path = load_path + "/" + str(models_paths[column_idx])
            # model = tf.keras.models.load_model(path)
            model = tf.keras.models.load_model(self.location + "/" + models_paths[column_idx],
                                               custom_objects={"AdamWeightDecay": AdamW})

            if type(model.optimizer) == ak.keras_layers.AdamWeightDecay:
                model.compile(optimizer='adam', loss='mse')

            y_train, y_val = [get_partially_train_data(data, c_name, column_type_separation) for data in output_data]

            # print(f'Training {models_paths[column_idx]} for additional {epochs} epochs')
            model.fit(train_X, y_train, validation_data=(val_X, y_val),
                      epochs=epochs, callbacks=[es_callback, lr_callback], verbose=0)
            model.save(self.location + "/" + models_paths[column_idx])

        end_time = datetime.now()
        diff = end_time - start_time
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Additional training finished".format(end_time.strftime("%d-%m-%Y %H-%M")) + "\n")
            f.write("       Elapsed time: {}".format(diff) + "\n")

    def _find_architecture(self,
                           input_data,
                           output_data,
                           target_columns,
                           column_type_separation,
                           max_model_size=None,
                           max_trials=150,
                           epochs=1000,
                           project_name='AutoAutoML',
                           save_path=""):
        # Launches autokeras search for the best architecture
        self.progress["current_task"] = "architecture_finding"

        train_X, val_X = input_data
        train_y, val_y = output_data

        models_path = []
        t = datetime.now()
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Finding architecture started".format(t.strftime("%d-%m-%Y %H-%M")) + "\n")

        for column_idx, c_name in tzip(range(len(target_columns)), target_columns, desc="Finding architecture",
                                       leave=True):
            start_time = datetime.now()
            with open("{}/log.txt".format(self.location), "a") as f:
                f.write(
                    "           {}: param '{}' searching".format(start_time.strftime("%d-%m-%Y %H-%M"), c_name) + "\n")

            y_train, y_val = [get_partially_train_data(data, c_name, column_type_separation) for data in output_data]

            linear_dims = train_X[0].shape[-1]
            cat_dims = [df.shape[-1] for df in train_X[1:]]
            input_cfg = [linear_dims, cat_dims]

            output_cfg = [df.shape[-1] for df in y_train]
            # tasks = []
            # for name in c_name:
            #     if name in column_type_separation["linear"]:
            #         tasks.append("regression")
            #     elif name in column_type_separation["categorical"]:
            #         tasks.append("classification")

            tasks = ["regression"] + ["classification"] * (len(y_train) - 1)

            inputs, outputs = build_ak_fnn1d_model(input_cfg, output_cfg, tasks=tasks)

            # print('\nTraining ', project_name + '_' + str(target_columns[column_idx]))
            trial_num = 0
            start_time = datetime.now()

            pn = str(target_columns[column_idx])
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

            # print("finding")
            model.fit(train_X, y_train,
                      validation_data=(val_X, y_val),
                      epochs=epochs, verbose=0,
                      callbacks=cb_list,
                      batch_size=64)

            # print("here")

            tf_model = model.export_model()
            path = "temp/" + project_name.split("/")[0] + f"/{target_columns[column_idx]}"
            # path = f'{save_path}/tf_model_{project_name}_{target_columns[column_idx]}'
            tf_model.save(self.location + "/" + path)
            # models_path.append(f'tf_model_{project_name}_{target_columns[column_idx]}')
            models_path.append(path)
            end_time = datetime.now()
            diff = end_time - start_time
            with open("{}/log.txt".format(self.location), "a") as f:
                f.write("           {}: param '{}' search finished".format(end_time.strftime("%d-%m-%Y %H-%M"),
                                                                           c_name) + "\n")
                f.write("           Elapsed time: {}".format(diff) + "\n")

        e_time = datetime.now()
        with open("{}/log.txt".format(self.location), "a") as f:
            f.write("       {}: Finding architecture finished".format(e_time.strftime("%d-%m-%Y %H-%M")) + "\n")
            f.write("       Elapsed time: {}".format(e_time - t) + "\n")
        return models_path

    def _get_models(self, input_data,
                    output_data,
                    target_columns,
                    column_type_separation,
                    max_model_size=None,
                    max_trials=150,
                    epochs=1000,
                    project_name='AutoAutoML',
                    additional_training_epochs=10000,
                    n_models=6,
                    save_path=""):

        models_paths = self._find_architecture(input_data=input_data,
                                               output_data=output_data,
                                               target_columns=target_columns,
                                               column_type_separation=column_type_separation,
                                               max_model_size=max_model_size,
                                               max_trials=max_trials, epochs=epochs,
                                               project_name=project_name,
                                               save_path=save_path)

        self._train_some_more(input_data=input_data,
                              output_data=output_data,
                              target_columns=target_columns,
                              column_type_separation=column_type_separation,
                              models_paths=models_paths,
                              epochs=additional_training_epochs,
                              load_path=save_path)

        self._train_uncertainty_models(input_data=input_data,
                                       output_data=output_data,
                                       target_columns=target_columns,
                                       column_type_separation=column_type_separation,
                                       model_paths=models_paths,
                                       n_models=n_models,
                                       epochs=additional_training_epochs,
                                       load_path=save_path)

        return models_paths

    def _get_models_with_good_enough_uncertainty(self,
                                                 input_data, output_data,
                                                 target_columns,
                                                 column_type_separation,
                                                 max_model_size=None,
                                                 max_trials=1,
                                                 epochs=1,
                                                 project_name='AutoAutoML',
                                                 additional_training_epochs=1,
                                                 uncertainty_threshold=0.05,
                                                 uncertain_points_threshold=0,
                                                 uncertain_combinations_threshold=0.2,
                                                 n_attempts=1,
                                                 save_path=""):

        train_X, val_X = input_data
        train_y, val_y = output_data
        output_data_copy = [df.copy() for df in output_data]

        n_models_found = 0
        found_models = [False for _ in range(len(target_columns))]
        best_uncertainties = [np.inf for _ in range(len(target_columns))]
        final_model_names = ['' for _ in range(len(target_columns))]
        cur_attempt = 0
        cur_target_columns = target_columns.copy()

        for cur_attempt in tqdm(range(n_attempts), desc="Attempt {}/{}".format(cur_attempt, n_attempts), leave=False):
            if n_models_found == len(target_columns): break

            self.progress["current_attempt"] = cur_attempt
            self.progress["n_models_found"] = n_models_found

            # while n_models_found < len(target_columns) and cur_attempt < n_attempts:
            # cur_attempt += 1
            start_time = datetime.now()
            with open("{}/log.txt".format(self.location), "a") as f:
                f.write(
                    "   {}: Attempt #{} calculating".format(start_time.strftime("%d-%m-%Y %H-%M"), cur_attempt) + "\n")

            cur_project_name = f'attempt_{cur_attempt}/' + project_name
            models_paths = self._get_models(input_data=input_data,
                                            output_data=output_data_copy,
                                            target_columns=cur_target_columns,
                                            column_type_separation=column_type_separation,
                                            max_model_size=max_model_size,
                                            max_trials=max_trials,
                                            epochs=epochs,
                                            project_name=cur_project_name,
                                            additional_training_epochs=additional_training_epochs,
                                            n_models=self.n_uncertainty_models,
                                            save_path=save_path)
            # TODO: add param name
            t = datetime.now()
            with open("{}/log.txt".format(self.location), "a") as f:
                f.write("       {}: Uncertainty calculating".format(t.strftime("%d-%m-%Y %H-%M")) + "\n")

            if n_attempts == 1:
                final_model_names = models_paths
            else:
                times = []
                for idx, model_found in enumerate(found_models):
                    if model_found:
                        continue

                    unc_stime = datetime.now()
                    input_df = pd.concat(train_X, axis=1)
                    if self.check_uncertainty and self.uncertainty_checker:
                        suffix = [""] + ["_unc_{}".format(i) for i in range(self.n_uncertainty_models)]
                        temp_path = [models_paths[idx] + s for s in suffix]
                        average_pred, preds = self._predict_uncertainty(temp_path, X=train_X)

                        param_name = models_paths[idx].split("/")[-1]
                        self.progress["current_task"] = "uncertainty_calculation"

                        is_good, uncertain_combs_count = self.uncertainty_checker(input_data=input_data,
                                                                                  output_data=output_data,
                                                                                  predictions=preds,
                                                                                  uncertainty_threshold=uncertainty_threshold,
                                                                                  params_to_combine=list(
                                                                                      input_df.columns),
                                                                                  param_name=param_name,
                                                                                  uncertain_combinations_threshold=uncertain_combinations_threshold,
                                                                                  uncertain_points_threshold=uncertain_points_threshold
                                                                                  )
                    else:
                        raise NotImplementedError()

                    if is_good:
                        found_models[idx] = True
                        n_models_found += 1
                        self.progress["n_models_found"] = n_models_found
                        cur_target_columns.remove(target_columns[idx])
                        final_model_names[idx] = models_paths[idx]

                    else:
                        if best_uncertainties[idx] > uncertain_combs_count:
                            best_uncertainties[idx] = uncertain_combs_count
                            final_model_names[idx] = models_paths[idx]
                    times.append(str(datetime.now() - unc_stime))

                with open("{}/log.txt".format(self.location), "a") as f:
                    f.write(
                        "       {}: Uncertainty calculating finished".format(
                            datetime.now().strftime("%d-%m-%Y %H-%M")) + "\n")
                    f.write("       Elapsed time: {}".format(datetime.now() - t) + "\n")
                    f.write("       " + " | ".join(times) + "\n")

            end_time = datetime.now()
            diff = end_time - start_time

            with open("{}/log.txt".format(self.location), "a") as f:
                f.write("   {}: Attempt #{} finished".format(end_time.strftime("%d-%m-%Y %H-%M"), cur_attempt) + "\n")
                f.write("   Elapsed time: {}".format(diff) + "\n")

        return final_model_names


def get_partially_train_data(Y, c_name, column_type_separation):
    linear_targets = [item for item in c_name if item in column_type_separation["linear"]]
    categorical_targets = [item for item in c_name if item in column_type_separation["categorical"]]
    linear_df = Y[0][linear_targets] if linear_targets else None
    if categorical_targets:
        dfs = Y[1:] if linear_targets else Y
        categorical_dfs = get_categoricals(dfs, categorical_targets)
    else:
        categorical_dfs = []

    reduced_data = [linear_df, *categorical_dfs]
    result = [item for item in reduced_data if item is not None]
    return result


def get_categoricals(data, targets):
    dfs_columns = [df.columns for df in data]
    target_dfs_idxs = []

    for target_name in targets:
        for idx, columns in enumerate(dfs_columns):
            if target_name in columns[0]:
                target_dfs_idxs.append(idx)

    categorical_dfs = [data[i] for i in target_dfs_idxs]
    return categorical_dfs


def split_array(arr):
    return [arr[:, i].reshape(-1, 1) for i in range(arr.shape[-1])]


if __name__ == '__main__':
    instance = Ensemble()
