from tensorflow.python.keras.models import Model


# TODO: implement model assumption to speed-up prediction, implement data processing

class PredictionWrapper1D:
    def __init__(self, models):
        self.models = models

    def predict(self, data, **kwargs):
        res = data.copy()
        for model in self.models:
            res = model.predict(res, **kwargs)

        return res

    def _preprocess(self, data):
        pass

    def _postprocess(self):
        pass
