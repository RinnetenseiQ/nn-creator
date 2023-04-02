import os

from nn_creator_core.models.custom_layers import UncertaintyLayer, EnsembleAverage
from nn_creator_core.utils.utils import default_kwargs
from tensorflow.python.keras.layers import Layer, average, Softmax
from tensorflow.python.keras.models import save_model, load_model, Model
import tensorflow as tf


def create_unc_model(models, cfg):
    inputs = models[0].inputs
    # for idx, model in enumerate(models):
    #     model._name = "{}_{}".format(model._name, idx)
    x = [model(inputs) for model in models]
    outputs = UncertaintyLayer(cfg)(x)
    return Model(inputs=inputs, outputs=outputs)


def assemble_model(models, cfg):
    if len(models) > 1:
        inputs = models[0].inputs
        for idx, model in enumerate(models):
            model._name = "{}_{}".format(model._name, idx)
        x = [model(inputs) for model in models]
        if type(x[0]) != list:
            x = [[item] for item in x]
        outputs = EnsembleAverage(cfg)(x)
        model_f = Model(inputs=inputs, outputs=outputs)
        return model_f
    elif len(models) == 1:
        return models[0]
    else:
        raise ValueError()


def create_fit_model(models, cfg):
    if len(models) > 1:
        inputs = models[0].inputs
        for idx, model in enumerate(models):
            model._name = "{}_{}".format(model._name, idx)
        x = [model(inputs) for model in models]
        outputs = [out for outs in x for out in outs]
        return Model(inputs=inputs, outputs=outputs)
    elif len(models) == 1:
        return models[0]
    else:
        raise ValueError()


class UncWrapper:
    @default_kwargs(is_seq=False)
    def __init__(self, models=None, cfg=None, **kwargs):
        self.models = models if models else []
        self._model = None
        self._trainable = True
        self.cfg = cfg if cfg else []
        self.is_seq = kwargs["is_seq"]
        if not self.is_seq and self.models:
            lin_losses = ["mean_squared_error"] if cfg[0] else []
            cat_losses = ["categorical_crossentropy"] * len(cfg[1]) if cfg[1] else []
            losses = lin_losses + cat_losses
            for model in self.models:
                model.loss = losses

    @property
    def model(self):
        if not self._model:
            self._model = assemble_model(self.models, self.cfg)
        return self._model

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        for model in self.models:
            model.trainable = value

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                jit_compile=None,
                **kwargs
                ):
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss,
                          metrics=metrics, weighted_metrics=weighted_metrics,
                          loss_weights=loss_weights)

        if not self.is_seq:
            self._model = assemble_model(self.models, self.cfg)
            # fit_model = create_fit_model(self.models, self.cfg)
            # fit_model.compile(optimizer=optimizer, loss=loss * len(self.models),
            #                   metrics=metrics, weighted_metrics=weighted_metrics,
            #                   loss_weights=loss_weights * len(self.models))
            self._model.compile(optimizer=optimizer, loss=loss,
                                metrics=metrics, weighted_metrics=weighted_metrics,
                                loss_weights=loss_weights)

    def fit(self, X, y,
            epochs=1,
            validation_data=None,
            callbacks=None, shuffle=True,
            verbose="auto", **kwargs):
        histories = []
        if self.is_seq:
            for model in self.models:
                history = model.fit(X, y, epochs=epochs,
                                    validation_data=validation_data,
                                    callbacks=callbacks,
                                    shuffle=shuffle,
                                    verbose=verbose)
                histories.append(history.history)
        else:

            history = self._model.fit(X, y, epochs=epochs,
                                      validation_data=validation_data,
                                      callbacks=callbacks,
                                      shuffle=shuffle,
                                      verbose=verbose)
            histories.append(history.history)
        return histories

    def predict(self, X):
        if self.is_seq:
            raise NotImplementedError()
        else:
            return self._model.predict(X)

    def check_uncertainty(self, X):
        unc_model = create_unc_model(self.models, self.cfg)
        return unc_model.predict(X)

    def dump(self, path):
        os.makedirs(path, exist_ok=True)
        for idx, model in enumerate(self.models):
            save_model(model, "{}/model_{}.h5".format(path, idx))

    def load(self, path, cfg, custom_objects=None):
        for model_path in [f.path for f in os.scandir(path)]:
            model = load_model(model_path, custom_objects=custom_objects)
            self.models.append(model)
        self._model = assemble_model(self.models, cfg)
        self.cfg = cfg
        return self


if __name__ == "__main__":
    # m = UncWrapper()
    # em =
    print("")
