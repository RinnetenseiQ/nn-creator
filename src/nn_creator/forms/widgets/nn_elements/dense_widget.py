import sys
from uuid import uuid4

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from nn_creator.forms.widgets.base_classes import BaseNNWidget
from nn_creator.forms.widgets.test_frame import TestFrame


class DenseWidget(BaseNNWidget):

    def __init__(self,
                 parent=None,
                 widget_id=None,
                 position=None,
                 event_filter=None):
        widget_size = (120, 60)
        pixmap = QPixmap("data/resources/nn_elements/dense.png")
        super().__init__(parent=parent, pixmap=pixmap, widget_id=widget_id, position=position, size=widget_size, event_filter=event_filter)
        self.setFixedSize(*widget_size)

        name = "dense_{}".format(uuid4())
        self.cfg = {'class_name': 'Dense',
                    'config': {
                        # 'name': 'dense_22',
                        # 'trainable': True,
                        # 'dtype': 'float32',
                        'units': 10,
                        'activation': 'linear',
                        'use_bias': True,
                        # 'kernel_initializer': {'class_name': 'GlorotUniform',
                        #                        'config': {'seed': None}},
                        # 'bias_initializer': {'class_name': 'Zeros', 'config': {}},
                        # 'kernel_regularizer': None,
                        # 'bias_regularizer': None,
                        # 'activity_regularizer': None,
                        # 'kernel_constraint': None,
                        # 'bias_constraint': None
                    },
                    'name': name,
                    'inbound_nodes': [[['input_16', 0, 0, {}]]]}

    def get_config(self):
        return self.cfg

    def connect(self, widgets: list[BaseNNWidget]):
        assert len(widgets) == 1
        name = widgets[0].cfg["name"]
        self.cfg['inbound_nodes'] = [[[name, 0, 0, {}]]]

    def set_config(self, units, use_bias=True):
        self.cfg["config"]["units"] = units
        self.cfg["config"]["use_bias"] = use_bias

    def mousePressEvent(self, event):
        super(DenseWidget, self).mousePressEvent(event)
        if self.is_connection_mode:
            self.connect([self.parent().connection_mode_widget])



if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = DenseWidget(position=(10, 10))
    window = QMainWindow()
    frame = TestFrame(widgets=[widget])
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())
