import sys

from PyQt5.QtCore import QSize, QDataStream, QIODevice, QPoint, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt


class CustomWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedSize(30, 30)
        self.update()


    def paintEvent(self, event: QPaintEvent) -> None:
        print("paint")
        painter = QPainter(self)
        pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")  # указываем путь к .png файлу
        painter.drawPixmap(self.rect(), pixmap)
        painter.end()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-dnditemdata'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        data = event.mimeData().data('application/x-dnditemdata')
        stream = QDataStream(data, QIODevice.ReadOnly)
        pixmap = QPixmap()
        offset = QPoint()
        stream >> pixmap >> offset

        new_position = event.pos() - offset
        self.move(new_position)
        event.setDropAction(Qt.MoveAction)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            mime_data = QMimeData()
            pixmap = self.grab()
            mime_data.setData('application/x-dnditemdata', pixmap)

            drag = QDrag(self)
            drag.setMimeData(mime_data)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.setPixmap(pixmap)

            drop_action = drag.exec_(Qt.MoveAction)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    window2.layout().addWidget(CustomWidget())
    window2.show()
    sys.exit(app.exec_())