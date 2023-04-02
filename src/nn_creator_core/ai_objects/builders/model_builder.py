from collections import defaultdict

from nn_creator_core.ai_objects.builders.ae1d import build_std_ae1d_model
from nn_creator_core.ai_objects.builders.vae1d import build_std_vae1d_model
from nn_creator_core.models.custom_layers import OneHotDecoder
from nn_creator_core.models.uncwrapper import UncWrapper
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.layers import average, concatenate, Softmax, Layer
from tensorflow.python.keras.models import Model

BUILDERS = {"vae": build_std_vae1d_model,
            "ae": build_std_ae1d_model}


def assemble_main_model(base_model, cfg):
    cont_len, dum_lens = cfg[0]
    inputs = base_model.inputs
    x = base_model(inputs)
    if cont_len and dum_lens:
        dumm_outputs = OneHotDecoder()(x[1:])
        output = [x[0], concatenate(dumm_outputs, axis=-1)]
        # output = concatenate([x[0]] + dumm_outputs, axis=-1)
    elif cont_len:
        output = [to_list(x)]
        # output = [x]
    elif dum_lens:
        output = [to_list(OneHotDecoder()(x))]
    else:
        raise ValueError()
    model = Model(inputs=inputs, outputs=output, name="main_model")
    return model


class ModelBuilderAE1D:
    def __init__(self, model_type, cfg, uncertainty_arg=None):
        uncertainty_arg = {} if not uncertainty_arg else uncertainty_arg
        self.uncertainty_arg = uncertainty_arg
        self.cfg = cfg
        assert model_type in ["vae", "ae"]
        self.model_type = model_type

    def build(self, **kwargs):
        if self.uncertainty_arg:
            components = defaultdict(list)
            for _ in range(self.uncertainty_arg["n_models"]):
                parts, trainable, model = BUILDERS[self.model_type](*self.cfg)

                encoder, decoder, imitator = parts
                endecoder, imdecoder = trainable

                components["encoders"].append(encoder)
                components["decoders"].append(decoder)
                components["imitators"].append(imitator)
                components["endecoders"].append(endecoder)
                components["imdecoders"].append(imdecoder)
                components["models"].append(model)

            # model = create_average_module(components["models"], "model", "model")
            imdecoder = UncWrapper(components["imdecoders"], cfg=self.cfg[0])
            # model = assemble_main_model(imdecoder.model, self.cfg)

            decoder = UncWrapper(components["decoders"])
            encoder = UncWrapper(components["encoders"])

            imitator = UncWrapper(components["imitators"])

            endecoder = UncWrapper(components["endecoders"], cfg=self.cfg[0])

            parts = [encoder, decoder, imitator]
            trainable = [endecoder, imdecoder]

            model = None

        else:
            parts, trainable, model = BUILDERS[self.model_type](*self.cfg)

        return parts, trainable, model


if __name__ == "__main__":
    builder = ModelBuilderAE1D("ae", [[5, [3, 4]], [4, [2, 3]], 3, 'he_uniform'], {"n_models": 5})
    parts, trainable, model = builder.build()
    encoder, decoder, imitator = parts
    endecoder, imdecoder = trainable
    m = imdecoder.models
    m[0].loss = ['mean_squared_error', 'categorical_crossentropy', 'categorical_crossentropy']
    m[0].compile(optimizer='adam',
                 loss=None,  # ['mean_squared_error', 'categorical_crossentropy', 'categorical_crossentropy'],
                 loss_weights=[1, 1, 1])
    m[1].compile(optimizer='adam',
                 loss=['mean_squared_error', 'categorical_crossentropy', 'categorical_crossentropy'],
                 loss_weights=[1, 1, 1])
    print("")
