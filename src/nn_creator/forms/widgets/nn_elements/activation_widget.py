from uuid import uuid4

from PyQt5.QtGui import QPixmap
from nn_creator.forms.widgets.base_classes import BaseNNWidget


class ActivationWidget(BaseNNWidget):

    def __init__(self,
                 parent=None,
                 widget_id=None,
                 position=None):
        widget_size = (125, 65)
        pixmap = QPixmap("data/resources/nn_elements/activation.png")
        super().__init__(parent=parent, pixmap=pixmap, widget_id=widget_id, position=position, size=widget_size)
        self.setFixedSize(*widget_size)

        name = "activation_{}".format(uuid4())
        self.cfg = {'class_name': 'Activation',
                    'config': {
                        # 'name': 'activation_3',
                        # 'trainable': True,
                        # 'dtype': 'float32',
                        'activation': 'linear'
                    },
                    'name': name,
                    'inbound_nodes': []}

    def get_config(self):
        return self.cfg

    def connect(self, widgets: list[BaseNNWidget]):
        assert len(widgets) == 1
        name = widgets[0].cfg["name"]
        self.cfg['inbound_nodes'] = [[[name, 0, 0, {}]]]

    def set_config(self, activation):
        self.cfg["config"]["activation"] = activation

