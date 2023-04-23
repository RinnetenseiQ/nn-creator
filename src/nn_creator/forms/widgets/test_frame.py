import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QMenu
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from nn_creator.forms.widgets.nn_elements.base_class import BaseNNWidget


class TestFrame(QFrame):
    def __init__(self, widgets: list[BaseNNWidget]):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        for idx, widget in enumerate(widgets):
            widget.setParent(self)
            widget.widget_id = idx
            widget.move(widget.position)

        ids = list(range(len(widgets)))
        it = list(zip(ids, widgets))
        self.widgets = dict(it)

        for widget in self.widgets.values():
            widget.cast_id_signal.connect(self.set_moved_widget_id)

        self.moved_widget_id = None
        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def dragEnterEvent(self, e):
        print("drag")
        # self.drag_widget = self.sender()
        e.accept()

    def dropEvent(self, e):
        print("drop")
        position = e.pos()

        widget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())
        widget.move(new_point)
        widget.show()
        widget.update()

        self.moved_widget_id = None
        e.setDropAction(Qt.MoveAction)
        e.accept()

    def set_moved_widget_id(self, widget_id):
        self.moved_widget_id = widget_id