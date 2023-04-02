import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, concatenate

from solver.master.models.custom_layers import OneHotDecoder, DenseBlock_small
import autokeras as ak
from autokeras import DenseBlock, Merge, StructuredDataInput

def build_ak_fnn1d_model(input_cfg, output_cfg, tasks=("regression",)):
    # output_cfg = [3] / [2, [2, 3]]
    def categorical_module_input(dum_input, shape):
        # x = DenseBlock(num_layers=1,
        #                num_units=shape * 2)(dum_input)
        #
        # x = DenseBlock(num_layers=1,
        #                num_units=shape)(x)
        x = DenseBlock()(dum_input)
        return x

    linear_input = StructuredDataInput()
    # cont_hidden = DenseBlock(num_layers=1,
    #                          num_units=10)(linear_input)
    cont_hidden = DenseBlock()(linear_input)

    dum_inputs = [StructuredDataInput() for _ in input_cfg[1]]
    dum_hidden = [categorical_module_input(inp, s) for inp, s in zip(dum_inputs, input_cfg[1])]

    merge = Merge(merge_type="concatenate")([cont_hidden] + dum_hidden)
    hidden = DenseBlock_small()(merge)

    outputs = []
    for task, output_dim in zip(tasks, output_cfg):
        hidden = DenseBlock(num_layers=1,
                            num_units=10)(hidden)
        if task == "regression":
            output = ak.RegressionHead(output_dim=output_dim)(hidden)
        elif task == "classification":
            output = ak.ClassificationHead(output_dim=output_dim)(hidden)
        else:
            raise NotImplementedError()
        outputs.append(output)

    inputs = [linear_input, *dum_inputs]
    # model2 = ak.AutoModel(inputs=[linear_input, *dum_inputs], outputs=linear_output, max_trials=2)
    return inputs, outputs


def build_std_fnn1d_model(input_cfg, output_cfg):
    def categorical_module_input(dum_input):
        x = Dense(dum_input.shape[-1] * 2, activation="relu")(dum_input)
        x = Dense(dum_input.shape[-1], activation="relu")(x)
        return x

    cont_len, dum_lens = input_cfg

    if cont_len:
        cont_input = Input(shape=(cont_len,))
        cont_hidden = Dense(cont_len * 2, activation="relu")(cont_input)
        cont_hidden = Dense(cont_len, activation="relu")(cont_hidden)
        cont_hidden = [cont_hidden]
        cont_input = [cont_input]
    else:
        cont_input = []
        cont_hidden = []

    if dum_lens:
        dum_inputs = [Input(shape=s) for s in dum_lens]
        dum_hidden = [categorical_module_input(inp) for inp in dum_inputs]
    else:
        dum_inputs = []
        dum_hidden = []

    to_merge = cont_hidden + dum_hidden
    merge = concatenate(to_merge, axis=-1)
    n = np.sum([cont_len, *dum_lens])
    hidden = Dense(n, activation="relu")(merge)
    hidden = Dense(n, activation="relu")(hidden)

    cont_len, dum_lens = output_cfg

    if cont_len:
        cont_output = Dense(cont_len, activation="linear")(hidden)
        cont_output = [cont_output]
    else:
        cont_output = []

    if dum_lens:
        dum_outputs = [Dense(length, activation="softmax")(hidden) for length in dum_lens]
    else:
        dum_outputs = []

    inputs = cont_input + dum_inputs
    outputs = cont_output + dum_outputs
    trainable = Model(inputs=inputs, outputs=outputs)

    x = trainable(trainable.inputs)
    if cont_len and dum_lens:
        dumm_outputs = OneHotDecoder()(x[1:])
        output = [x[0], concatenate(dumm_outputs, axis=-1)]
        # output = concatenate([x[0]] + dumm_outputs, axis=-1)
    elif cont_len:
        output = x
    elif dum_lens:
        output = OneHotDecoder()(x)
    else:
        raise ValueError()

    main_model = Model(inputs=trainable.inputs, outputs=output)
    return trainable, main_model


if __name__ == '__main__':
    cont_len = 10
    dum_lens = [5, 6]

    input_cfg = [cont_len, dum_lens]

    cont_len = 5
    dum_lens = [3, 4]

    output_cfg = [cont_len, dum_lens]

    trainable, model = build_std_fnn1d_model(input_cfg, output_cfg)

    d = model.get_config()
    layer = dict(d["layers"][0])
    d["layers"][0] = layer
    restored = Model().from_config(model.get_config(), custom_objects={"OneHotDecoder": OneHotDecoder})
    print("")
