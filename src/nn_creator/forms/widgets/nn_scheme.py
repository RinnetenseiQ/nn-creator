import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QWidget
from nn_creator.forms.utils.event_filters import GlobalEventFilter2
from nn_creator.forms.widgets.base_classes import BaseNNWidget
from nn_creator.forms.widgets.tests.connection import ConnectionWidget


class CursorPos(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        pos = self.window.mapFromGlobal(self.cursor().pos())
        x = pos.x()
        y = pos.y()
        print(f"Cursor position: x={x}, y={y}")


class NNSchemeWidget(QFrame):
    set_moved_widget_id_signal = pyqtSignal(int)

    def __init__(self, parent=None, event_filter: GlobalEventFilter2 = None):
        super().__init__(parent=parent)
        self._event_filter = event_filter
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    @property
    def event_filter(self):
        return self._event_filter

    @event_filter.setter
    def event_filter(self, value: GlobalEventFilter2):
        self._event_filter = value

    def dragEnterEvent(self, e):
        # self.updateCursorPosition()
        # self.drag_widget = self.sender()
        e.accept()

    def dropEvent(self, e):
        position = e.pos()
        if self.event_filter.nn_scheme_moved_widget_id < 0: return
        widget = self.event_filter.nn_scheme_widgets[self.event_filter.nn_scheme_moved_widget_id]

        # widget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())

        widget.move(new_point)
        widget.show()
        widget.update()

        self.set_moved_widget_id_signal.emit(-1)
        e.setDropAction(Qt.MoveAction)
        e.accept()

    # @pyqtSlot()
    def update_children(self, child: QWidget):
        child.setParent(self)
        if isinstance(child, ConnectionWidget):
            child.set_paint_mode(True)
            print(child.geometry())
            print(self.geometry())
            child.show()

        # child.show()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
