from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, concatenate, Layer, Lambda
from tensorflow.python.keras.backend import square, exp, random_normal
from tensorflow import shape, reduce_mean, reduce_sum, add, multiply
from tensorflow.keras.utils import plot_model

from solver.master.models.custom_layers import OneHotDecoder, Sampling


def kl_loss(z_mean, z_log_var):
    kl_lss = 1 + z_log_var - square(z_mean) - exp(z_log_var)
    kl_lss = reduce_mean(kl_lss, axis=-1)
    kl_lss *= -0.5
    return kl_lss


def build_encoder(cont_len, dum_lens, output_size, initializer='glorot_uniform'):
    def categorical_module_input(dum_input):
        x = Dense(dum_input.shape[-1] * 2, activation="relu", kernel_initializer=initializer)(dum_input)
        x = Dense(dum_input.shape[-1], activation="relu", kernel_initializer=initializer)(x)
        return x

    if cont_len:
        cont_input = Input(shape=(cont_len,))
        cont_hidden = Dense(cont_len * 2, activation="relu", kernel_initializer=initializer)(cont_input)
        cont_hidden = Dense(cont_len, activation="relu", kernel_initializer=initializer)(cont_hidden)
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

    hid = cont_hidden + dum_hidden
    merge = concatenate(hid, axis=-1) if len(hid) > 1 else hid[0]

    output_mean = Dense(output_size, activation="relu", kernel_initializer=initializer)(merge)
    output_log_var = Dense(output_size, activation="relu", kernel_initializer=initializer)(merge)
    output_sample, output_mean, output_log_var = Sampling()((output_mean, output_log_var))

    inputs = cont_input + dum_inputs
    encoder = Model(inputs=inputs, outputs=[output_mean, output_log_var, output_sample])
    return encoder


def build_decoder(cont_len, dum_lens, input_size, initializer='glorot_uniform'):
    dec_input = Input(shape=(input_size,))
    dec_hidden = Dense(input_size * 2, activation="relu", kernel_initializer=initializer)(dec_input)

    if cont_len:
        cont_output = Dense(cont_len, activation="linear", kernel_initializer=initializer)(dec_hidden)
        cont_output = [cont_output]
    else:
        cont_output = []

    if dum_lens:
        dum_outputs = [Dense(length, activation="softmax", kernel_initializer=initializer)(dec_hidden) for length in dum_lens]
    else:
        dum_outputs = []

    outputs = cont_output + dum_outputs

    return Model(inputs=[dec_input], outputs=outputs)


def build_imitator(cont_len, dum_lens, output_size, initializer="glorot_uniform"):
    def categorical_module_input(dum_input):
        x = Dense(dum_input.shape[-1] * 2, activation="relu", kernel_initializer=initializer)(dum_input)
        x = Dense(dum_input.shape[-1], activation="relu", kernel_initializer=initializer)(x)
        return x

    if cont_len:
        cont_input = Input(shape=(cont_len,))
        cont_hidden = Dense(cont_len * 2, activation="relu", kernel_initializer=initializer)(cont_input)
        cont_hidden = Dense(cont_len, activation="relu", kernel_initializer=initializer)(cont_hidden)
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

    hid = cont_hidden + dum_hidden
    merge = concatenate(hid, axis=-1) if len(hid) > 1 else hid[0]

    output = Dense(output_size, activation="relu", kernel_initializer=initializer)(merge)

    inputs = cont_input + dum_inputs
    imitator = Model(inputs=inputs, outputs=output)
    return imitator


def build_std_vae1d_model(encoder_cfg, imitator_cfg, encode_size, initializer='glorot_uniform'):
    cont_len, dum_lens = encoder_cfg

    encoder = build_encoder(cont_len, dum_lens, encode_size, initializer=initializer)
    decoder = build_decoder(cont_len, dum_lens, encode_size, initializer=initializer)

    y_mean, y_log_var, y_sample = encoder(encoder.inputs)
    endecoder_output = decoder(y_sample)
    endecoder = Model(encoder.inputs, endecoder_output, name='endecoder')
    endecoder.add_loss(reduce_mean(kl_loss(y_mean, y_log_var)) / ((cont_len + sum(dum_lens)) ** 2))

    cont_len, dum_lens = imitator_cfg
    imitator = build_imitator(cont_len, dum_lens, encode_size, initializer=initializer)

    x_sample = imitator(imitator.inputs)
    imdecoder_output = decoder(x_sample)
    imdecoder = Model(imitator.inputs, imdecoder_output, name='imdecoder')
    # imdecoder.add_loss(reduce_mean(kl_loss(x_mean, x_log_var))/((cont_len+sum(dum_lens))**2))

    cont_len, dum_lens = encoder_cfg
    x = imdecoder(imdecoder.inputs)
    if cont_len and dum_lens:
        dumm_outputs = OneHotDecoder()(x[1:])
        output = [x[0], concatenate(dumm_outputs, axis=-1)]
        # output = concatenate([x[0]] + dumm_outputs, axis=-1)
    elif cont_len:
        output = [x]
    elif dum_lens:
        output = [OneHotDecoder()(x)]
    else:
        raise ValueError()

    main_model = Model(inputs=imdecoder.inputs, outputs=output)

    return ((encoder, decoder, imitator),
            (endecoder, imdecoder),
            main_model)
