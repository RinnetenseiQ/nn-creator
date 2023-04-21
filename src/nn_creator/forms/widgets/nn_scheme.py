import sys

from PyQt5.QtCore import QSize, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QPushButton
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

from nn_creator.forms.widgets.nn_elements.add_widget import AddWidget


class NNSchemeWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        self.widgets = {}

        for widget in self.widgets.values():
            widget.cast_id_signal.connect(self.set_moved_widget_id)
            widget.delete_widget_signal.connect(self.delete_widget_id)

        self.moved_widget_id = None
        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def dragEnterEvent(self, e):
        print(f"drag id: {self.moved_widget_id}")
        # self.drag_widget = self.sender()
        e.accept()

    def dropEvent(self, e):
        position = e.pos()

        widget: AddWidget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())
        widget.move(new_point)
        widget.show()
        widget.update()

        print(f"drop id: {self.moved_widget_id}")
        self.moved_widget_id = None
        e.setDropAction(Qt.MoveAction)
        e.accept()



    def set_moved_widget_id(self, widget_id):
        print("set_moved_widget_id", widget_id)
        self.moved_widget_id = widget_id

    def update_widgets_holder(self, widget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
        widget.widget_id = key
        widget.setParent(self)
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        # self.moved_widget_id = key
        print(f"widgets_ids: {self.widgets.keys()}")

    def delete_widget_id(self, widget_id):
        print(f"delete: {widget_id}")
        self.widgets.pop(widget_id)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
