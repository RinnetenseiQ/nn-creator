import sys

from PyQt5.QtCore import QSize, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QPushButton
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np




class NNSchemeWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        self.widgets = {}

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

    def update_widgets_holder(self, widget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
        widget.widget_id = key
        widget.setParent(self)
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        # self.moved_widget_id = key


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
