import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent, QPoint
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QMenu
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from abc import abstractmethod


class BaseNNWidget(QWidget):
    delete_widget_signal = pyqtSignal(int)
    cast_id_signal = pyqtSignal(int)
    mouse_press_signal = pyqtSignal(int)

    def __init__(self, pixmap, parent=None, widget_id=None, position=(0, 0), size=(30, 30)):
        self.cfg = None
        super().__init__(parent=parent)
        self.widget_id = widget_id
        self.setFixedSize(*size)
        self.setAcceptDrops(True)
        position = position if position else (0, 0)
        self.position = QPoint(*position)
        # self.setDragEnabled(True)
        self._pixmap = pixmap.scaled(*size)
        if position:
            self.move(self.position)
        self.drag_start_position = self.pos()
        self.update()

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def connect(self, widgets):
        pass

    @property
    def pixmap(self):
        return self._pixmap

    def paintEvent(self, event: QPaintEvent) -> None:
        print("paint")
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._pixmap)
        painter.end()

    def mousePressEvent(self, event):
        print("mouse press")
        if event.button() == Qt.LeftButton:
            # Запоминаем позицию относительно виджета
            self.drag_start_position = event.pos()
            self.mouse_press_signal.emit(self.widget_id)

    def mouseMoveEvent(self, event):
        print("mouse move-")

        if event.buttons() == Qt.LeftButton:
            self.cast_id_signal.emit(self.widget_id)
            self.hide()
            self.update()
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(self._pixmap)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        print('mouseReleaseEvent')

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        print("leave event-")

    def dragLeaveEvent(self, a0: QtGui.QDragLeaveEvent) -> None:
        print("drag leave event")

    # def dropEvent(self, a0: QtGui.QDropEvent) -> None:
    #     print("add widget drop event")

    def minimumSizeHint(self) -> QtCore.QSize:
        return QSize(30, 30)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSizeHint()

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        contextMenu.setStyleSheet("background-color:blue;")
        delete_action = contextMenu.addAction("Delete")
        connect_action = contextMenu.addAction("Connect")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == delete_action:
            self.delete_widget_signal.emit(self.widget_id)
            self.close()
        if action == connect_action:
            pass