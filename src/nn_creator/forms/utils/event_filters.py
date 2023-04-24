import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent, pyqtSlot
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QMenu
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from nn_creator.forms.widgets.test_frame import TestFrame
import numpy as np


class GlobalEventFilter2(QObject):

    def __init__(self):
        super().__init__()
        self._nn_scheme_widgets: dict = {}
        self._nn_scheme_moved_widget_id: int = -1

    @property
    def nn_scheme_widgets(self):
        return self._nn_scheme_widgets

    # @pyqtSlot()
    def update_nn_scheme_widgets_list(self, widget):
        key = np.max(list(self.nn_scheme_widgets.keys())) + 1 if self.nn_scheme_widgets else 0
        widget.widget_id = key
        widget.cast_id_signal.connect(self.set_nn_widget_moved_widget_id)
        widget.delete_widget_signal.connect(self.delete_nn_scheme_widget)
        self.nn_scheme_widgets[key] = widget
        self.set_nn_widget_moved_widget_id(key)
        # self.moved_widget_id = key
        # print(f"event filter widgets_ids: {self.widgets.keys()}")

    # @pyqtSlot()
    def delete_nn_scheme_widget(self, widget_id: int):
        print(f"delete: {widget_id}")
        widget: QWidget = self.nn_scheme_widgets[widget_id]
        self.nn_scheme_widgets.pop(widget_id)
        widget.setParent(None)
        widget.close()

    @property
    def nn_scheme_moved_widget_id(self):
        return self._nn_scheme_moved_widget_id

    # @pyqtSlot()
    def set_nn_widget_moved_widget_id(self, widget_id: int):
        self._nn_scheme_moved_widget_id = widget_id

    def eventFilter(self, obj, event):
        # print("Event Filter: sum event happend")
        if event.type() == QEvent.MouseButtonRelease:
            if self.nn_scheme_widgets:
                self.nn_scheme_widgets[self.nn_scheme_moved_widget_id].show()
            print("Event Filter: Mouse Button Release")

        return super().eventFilter(obj, event)


class GlobalEventFilter(QObject):
    connection_mode_signal = pyqtSignal(bool)

    def __init__(self, widgets=None):
        super().__init__()
        self.widgets = widgets if widgets else {}
        self.moved_widget_id = None

    def eventFilter(self, obj, event):
        # print("Event Filter: sum event happend")
        if event.type() == QEvent.MouseButtonRelease:
            if self.widgets:
                self.widgets[self.moved_widget_id].show()
            print("Event Filter: Mouse Button Release")

        return super().eventFilter(obj, event)

    def set_moved_widget_id(self, widget_id):
        self.moved_widget_id = widget_id

    # @pyqtSlot()
    def update_widgets_list(self, widget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
        widget.widget_id = key
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        widget.delete_widget_signal.connect(self.delete_widget_id)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        # self.moved_widget_id = key
        print(f"event filter widgets_ids: {self.widgets.keys()}")

    def delete_widget_id(self, widget_id):
        print(f"delete: {widget_id}")
        self.widgets.pop(widget_id)
