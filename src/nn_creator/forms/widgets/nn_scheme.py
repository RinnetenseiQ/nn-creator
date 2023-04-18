import sys

from PyQt5.QtCore import QSize, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QPushButton
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt


class Button(QPushButton):

    def __init__(self, title, parent):
        super().__init__(title, parent)

    def mousePressEvent(self, event):
        print("mouse press")
        if event.button() == Qt.LeftButton:
            # Запоминаем позицию относительно виджета
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        print("mouse move")
        if event.buttons() == Qt.LeftButton:
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(self.grab())
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)


    # def mouseMoveEvent(self, e):
    #
    #     if e.buttons() != Qt.RightButton:
    #         return
    #
    #     mimeData = QMimeData()
    #
    #     drag = QDrag(self)
    #     drag.setMimeData(mimeData)
    #     drag.setHotSpot(e.pos() - self.rect().topLeft())
    #
    #     dropAction = drag.exec_(Qt.MoveAction)
    #
    #
    # def mousePressEvent(self, e):
    #
    #     QPushButton.mousePressEvent(self, e)
    #
    #     if e.button() == Qt.LeftButton:
    #         print('press')


class NNSchemeWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)
        self.button = Button(parent=self, title="Label")
        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        position = e.pos()
        self.button.move(position)

        e.setDropAction(Qt.MoveAction)
        e.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())