import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QMenu
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from nn_creator.forms.widgets.nn_elements.base_class import BaseNNWidget
from nn_creator.forms.widgets.test_frame import TestFrame
from uuid import uuid4


class AddWidget(BaseNNWidget):

    def __init__(self, parent=None, widget_id=None, position=None):
        widget_size = (30, 30)
        pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")
        super().__init__(parent=parent, pixmap=pixmap, widget_id=widget_id, position=position, size=widget_size)
        name = "add_{}".format(uuid4())
        self.cfg = {'class_name': 'Add',
                    'config': {
                        # 'name': 'add_1',
                        # 'trainable': True,
                        # 'dtype': 'float32'
                    },
                    'name': name,
                    'inbound_nodes': []}

    def get_config(self):
        return self.cfg

    def connect(self, widgets: list[BaseNNWidget]):
        assert len(widgets) > 1
        inbound = []
        for widget in widgets:
            name = widget.cfg["name"]
            inbound.append([name, 0, 0, {}])

        self.cfg['inbound_nodes'] = [inbound]




if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = AddWidget(position=(10, 10))
    window = QMainWindow()
    frame = TestFrame(widgets=[widget])
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())
