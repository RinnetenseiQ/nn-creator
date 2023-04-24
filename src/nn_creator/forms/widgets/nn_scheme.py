import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame
from nn_creator.forms.widgets.base_classes import BaseNNWidget
from nn_creator.forms.widgets.connection import ConnectionWidget


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

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)
        self.widgets = {}

        self.moved_widget_id = None
        self.is_connection_mode = False
        self.connection_mode_widget = None
        self.connections = {}
        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def dragEnterEvent(self, e):
        print(f"drag id: {self.moved_widget_id}")
        # self.updateCursorPosition()
        # self.drag_widget = self.sender()
        e.accept()

    def dropEvent(self, e):
        position = e.pos()

        widget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())
        self.check_geometry(child_widget=new_point, nn_scheme=self, window=self.parent())

        widget.move(new_point)
        widget.show()
        widget.update()

        print(f"drop id: {self.moved_widget_id}")
        self.moved_widget_id = None
        e.setDropAction(Qt.MoveAction)
        e.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        x = event.x()
        y = event.y()
        print(f"Cursor position: x={x}, y={y}")


    def check_geometry(self, child_widget, nn_scheme, window):
        child_geometry = child_widget
        nn_scheme_geometry = nn_scheme.geometry()
        window_geometry = window.geometry()
        print("widget: ", child_geometry, '\n', "scheme: ", nn_scheme_geometry, '\n', "window: ", window_geometry, '\n')
        # if not parent_geometry.contains(child_geometry):
        #     print("Child widget is outside parent widget bounds")

    def set_moved_widget_id(self, widget_id):
        print("set_moved_widget_id", widget_id)
        self.moved_widget_id = widget_id

    def update_widgets_holder(self, widget: BaseNNWidget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
        widget.widget_id = key
        widget.setParent(self)
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        widget.delete_widget_signal.connect(self.delete_widget_id)
        widget.connection_create_signal.connect(self.update_connections)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        # self.moved_widget_id = key
        print(f"widgets_ids: {self.widgets.keys()}")

    def delete_widget_id(self, widget_id):
        print(f"delete: {widget_id}")
        self.widgets[widget_id].close()
        self.widgets.pop(widget_id)

    def set_connection_mode(self, flag):
        self.is_connection_mode = flag

    def set_connection_mode_widget(self, widget: BaseNNWidget):
        self.connection_mode_widget = widget

    def update_connections(self, conn: ConnectionWidget):
        key = np.max(list(self.connections.keys())) + 1 if self.connections else 0
        conn.connection_id = key
        conn.setParent(self)
        conn.delete_signal.connect(self.delete_connection)
        self.connections[key] = conn

    def delete_connection(self, connection_id):
        self.connections[connection_id].close()
        self.connections.pop(connection_id)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
