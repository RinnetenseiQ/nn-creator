import sys

from PyQt5.QtCore import QSize, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QPushButton
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal


class NNSchemeWidget(QFrame):


    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)
        self.setStyleSheet("background-color:yellow;")

        self.widgets = []
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        position = e.pos()
        new_point = QtCore.QPoint(position.x() - self.widget.drag_start_position.x(),
                                  position.y() - self.widget.drag_start_position.y())

        e.setDropAction(Qt.MoveAction)
        e.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
