import sys

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt


class AddWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(30, 30)
        self.setAcceptDrops(True)
        # self.setDragEnabled(True)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        print("paint")
        painter = QPainter(self)
        pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")  # указываем путь к .png файлу
        painter.drawPixmap(self.rect(), pixmap)
        painter.end()

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

    # def dragEnterEvent(self, e):
    #     print("drag")
    #     if e.mimeData().hasUrls():
    #         e.acceptProposedAction()
    #     else:
    #         e.ignore()
    #     pass
    #
    # def dropEvent(self, e):
    #     print("drop")
    #     if e.mimeData().hasUrls():
    #         e.acceptProposedAction()
    #         e.accept()
    #         position = e.pos()
    #         print('Dropped at pos: ', position)
    #         # Дополнительные действия при перемещении виджета
    #         self.move(position)
    #         self.update()
    #     else:
    #         e.ignore()

    def minimumSizeHint(self) -> QtCore.QSize:
        return QSize(30, 30)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSizeHint()


class TestFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)
        self.widget = AddWidget(parent=self)
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
        self.widget.move(position)
        self.widget.update()
        e.setDropAction(Qt.MoveAction)
        e.accept()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    frame = TestFrame()
    window2.layout().addWidget(frame)
    window2.show()
    sys.exit(app.exec_())