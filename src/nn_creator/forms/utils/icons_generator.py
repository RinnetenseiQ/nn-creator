from nn_creator.forms.utils.resources import layers_icons


class ResourceGenerator:
    def __init__(self, theme):
        self.icon_layers = layers_icons[theme]


    def get_icon(self, type: str, name: str):
        if type == "layer":
            icon = self.icon_layers[name]
            label = name.replace("_", " ")
            return icon, label