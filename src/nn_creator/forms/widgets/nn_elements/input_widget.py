import sys
from uuid import uuid4

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from nn_creator.forms.widgets.base_classes import BaseNNWidget
from nn_creator.forms.widgets.test_frame import TestFrame


class InputWidget(BaseNNWidget):

    def __init__(self,
                 parent=None,
                 input_shape=None,
                 widget_id=None,
                 position=None,
                 event_filter=None):
        widget_size = (120, 60)
        pixmap = QPixmap("data/resources/nn_elements/input.png")
        super().__init__(parent=parent, pixmap=pixmap, widget_id=widget_id, position=position, size=widget_size, event_filter=event_filter)
        self.setFixedSize(*widget_size)

        shape = (None, *input_shape) if input_shape else (None, None)
        name = "input_{}".format(uuid4())
        self.cfg = {'class_name': 'InputLayer',
                    'config': {'batch_input_shape': shape,
                               'dtype': 'float32',
                               'sparse': False,
                               'ragged': False,
                               'name': name},
                    'name': 'input_4',
                    'inbound_nodes': []}

    def get_config(self):
        return self.config

    def connect(self, widgets: list[BaseNNWidget]):
        pass

    def mousePressEvent(self, event):
        super(InputWidget, self).mousePressEvent(event)
        if self.is_connection_mode:
            pass
            # tip = QToolTip()
            #
            # tip.showText(pos=event.pos(), text="can`t connect")

    def set_input_shape(self, shape):
        self.config["config"]['batch_input_shape'] = (None, *shape)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = InputWidget(position=(10, 10))
    window = QMainWindow()
    frame = TestFrame(widgets=[widget])
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())
