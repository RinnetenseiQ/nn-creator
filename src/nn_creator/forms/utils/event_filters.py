import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent, pyqtSlot
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QMenu
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt

from nn_creator.forms.widgets.nn_scheme import NNSchemeWidget
from nn_creator.forms.widgets.test_frame import TestFrame
import numpy as np
from nn_creator.forms.widgets.tests.connection import ConnectionWidget


class GlobalEventFilter2(QObject):
    connection_create_signal = pyqtSignal(ConnectionWidget)

    def __init__(self):
        super().__init__()
        self._nn_scheme_widgets: dict = {}
        self._nn_scheme_moved_widget_id: int = -1
        self._last_nn_scheme_deleted_widget_id: int = -1

        self._nn_scheme_connections: dict = {}

    @property
    def nn_scheme_widgets(self):
        return self._nn_scheme_widgets

    # @pyqtSlot()
    def update_nn_scheme_widgets_list(self, widget):
        key = np.max(list(self.nn_scheme_widgets.keys())) + 2 if self.nn_scheme_widgets else 1
        widget.widget_id = key
        widget.cast_id_signal.connect(self.set_nn_widget_moved_widget_id)
        widget.delete_widget_signal.connect(self.delete_nn_scheme_widget)
        widget.connection_create_signal.connect(self.add_connection)
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
        self._last_nn_scheme_deleted_widget_id = widget_id

    @property
    def nn_scheme_moved_widget_id(self):
        return self._nn_scheme_moved_widget_id

    # @pyqtSlot()
    def set_nn_widget_moved_widget_id(self, widget_id: int):
        self._nn_scheme_moved_widget_id = widget_id

    @property
    def nn_scheme_connections(self):
        return self._nn_scheme_connections

    def add_connection(self, conn):
        key = np.max(list(self.nn_scheme_widgets.keys())) + 1 if self.nn_scheme_widgets else 1
        conn.connection_id = key
        self.connection_create_signal.emit(conn)
        conn.delete_connection_signal.connect(self.delete_connection)
        self._nn_scheme_connections[key] = conn

    def delete_connection(self, connection_id: int):
        self._nn_scheme_connections.pop(connection_id)

    def eventFilter(self, obj, event):
        # print("Event Filter: sum event happend")
        if event.type() == QEvent.MouseButtonRelease:
            try:
                if self.nn_scheme_widgets:
                    if self._last_nn_scheme_deleted_widget_id != self.nn_scheme_moved_widget_id and self.nn_scheme_moved_widget_id > -1:
                        self.nn_scheme_widgets[self.nn_scheme_moved_widget_id].show()
            except KeyError as e:
                print(e)

            self._nn_scheme_moved_widget_id = -1
            print("Event Filter: Mouse Button Release")

        return super().eventFilter(obj, event)


class GlobalEventFilter(QObject):
    connection_mode_signal = pyqtSignal(bool)
    connection_create_signal = pyqtSignal(ConnectionWidget)

    def __init__(self, widgets=None):
        super().__init__()
        self.widget_holder = None
        self.widgets = widgets if widgets else {}
        self.moved_widget_id = None

        self._nn_scheme_widgets: dict = {}
        self._nn_scheme_connections: dict = {}
    def eventFilter(self, obj, event):
        # print("Event Filter: sum event happend")
        try:
            if event.type() == QEvent.MouseButtonRelease:
                if self.widgets:
                    if self.widget_holder.last_deleted_widget != self.moved_widget_id and self.moved_widget_id:
                        self.widgets[self.moved_widget_id].show()
                self.moved_widget_id = None

                print("Event Filter: Mouse Button Release", self.moved_widget_id)
        except KeyError as e:
            print(e)
        if event.type() == QEvent.MouseButtonPress:
            if type(obj) == NNSchemeWidget:
                print("This is NNSchemeWidget")

        return super().eventFilter(obj, event)

    def set_widget_holder(self, widget_holder):
        self.widget_holder = widget_holder
        self.widgets = self.widget_holder.widgets

    def created_widget(self, widget):
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        self.set_moved_widget_id(widget.widget_id)

    def set_moved_widget_id(self, widget_id):
        self.moved_widget_id = widget_id

    # @property
    # def nn_scheme_connections(self):
    #     return self._nn_scheme_connections
    #
    # def add_connection(self, conn):
    #     key = np.max(list(self._nn_scheme_widgets.keys())) + 1 if self._nn_scheme_widgets else 1
    #     conn.connection_id = key
    #     self.connection_create_signal.emit(conn)
    #     conn.delete_connection_signal.connect(self.delete_connection)
    #     self._nn_scheme_connections[key] = conn
    #
    # def delete_connection(self, connection_id: int):
    #     self._nn_scheme_connections.pop(connection_id)


