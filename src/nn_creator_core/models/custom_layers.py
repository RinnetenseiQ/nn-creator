import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from autokeras import StructuredDataInput, DenseBlock, Merge
import autokeras as ak
from tensorflow.python.keras.layers import Dense, Input, concatenate, average, Softmax
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.backend import square, exp, random_normal
from tensorflow import shape, reduce_mean, reduce_sum, add, multiply


class UncertaintyLayer(Layer):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg if cfg else []

    def get_config(self):
        config = super().get_config().copy()
        config.update({'cfg': self.cfg})
        return config

    def call(self, inputs, *args, **kwargs):
        if self.cfg:
            output = []
            if self.cfg[0]:
                lin_inps = []
                # cat_inps = []
                for inp in inputs:
                    lin_inps.append(inp[0])
                    # cat_inps.append(inp[1:])
                lin_out = tf.math.reduce_std(lin_inps, axis=0)
                output.append(lin_out)

            cat = []
            if self.cfg[1]:
                cat_inps = []
                for inp in inputs:
                    cat_inps.append(inp[1:])
                for idx in range(len(cat_inps[0])):
                    t = [item[idx] for item in cat_inps]
                    cat.append(tf.math.reduce_std(t, axis=0))
            output += cat
            return output


class EnsembleAverage(Layer):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg if cfg else []

    def get_config(self):
        config = super().get_config().copy()
        config.update({'cfg': self.cfg})
        return config

    def call(self, inputs, *args, **kwargs):
        if self.cfg:
            output = []
            if self.cfg[0]:
                lin_inps = []
                # cat_inps = []
                for inp in inputs:
                    lin_inps.append(inp[0])
                    # cat_inps.append(inp[1:])
                lin_out = average(lin_inps)
                output.append(lin_out)

            cat = []
            if self.cfg[1]:
                cat_inps = []
                for inp in inputs:
                    cat_inps.append(inp[1:])
                for idx in range(len(cat_inps[0])):
                    t = [item[idx] for item in cat_inps]
                    cat.append(Softmax()(average(t)))
            output += cat
            return output


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding"""

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        # batch = shape(z_mean)[0]
        # dim = shape(z_mean)[1]
        epsilon = random_normal(shape=shape(z_mean))
        return z_mean + exp(0.5 * z_log_var) * epsilon, z_mean, z_log_var


class OneHotDecoder(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def get_config(self):
    #     return {'k': self.k}

    def call(self, inputs, *args, **kwargs):
        dumm_outputs = [tf.math.argmax(output, axis=-1) for output in inputs]
        dumm_outputs = [tf.reshape(output, (-1, 1)) for output in dumm_outputs]
        dumm_outputs = [tf.cast(output, "float") for output in dumm_outputs]
        return dumm_outputs


def categorical_module_input(dum_input, shape):
    x = DenseBlock(num_layers=1,
                   num_units=shape * 2)(dum_input)

    x = DenseBlock(num_layers=1,
                   num_units=shape)(x)
    return x


class DenseBlock_small(ak.Block):
    def declare_hyperparameters(self, hp):
        pass

    def build(self, hp, inputs=None):
        max_layers = 6

        input_node = tf.nest.flatten(inputs)[0]
        output = input_node
        n_layers = hp.Int('num_layers', min_value=2, max_value=max_layers, step=1)
        activation = hp.Choice('activation', ['relu', 'elu'])  # , 'gelu'])
        use_dropout = hp.Boolean('use_dropout')
        # dropout_rate = hp.Float('Dropout_rate', min_value=0.01, max_value=0.4, sampling='log',
        #                             parent_name='use_dropout', parent_values=[True])
        dropout_rate = 0.1

        if activation == 'relu':
            activation_fn = tf.keras.layers.ReLU()
        elif activation == 'elu':
            activation_fn = tf.keras.layers.ELU()
        # elif activation == 'gelu':
        #     activation_fn = tf.keras.activations.gelu
        else:
            raise NotImplementedError()

        for i in range(n_layers):
            layer = tf.keras.layers.Dense(hp.Int(f'n_units_{i}', min_value=30, max_value=3000, step=20,
                                                 parent_name='num_layers',
                                                 parent_values=list(range(i + 1, max_layers + 1))))
            output = layer(output)
            output = activation_fn(output)
            if use_dropout:
                dropout = tf.keras.layers.Dropout(dropout_rate)
                output = dropout(output)

        return output


CUSTOM_OBJECTS = {"Sampling": Sampling,
                  "DenseBlock_small": DenseBlock_small,
                  "OneHotDecoder": OneHotDecoder,
                  "UncertaintyLayer": UncertaintyLayer,
                  "EnsembleAverage": EnsembleAverage
                  }
