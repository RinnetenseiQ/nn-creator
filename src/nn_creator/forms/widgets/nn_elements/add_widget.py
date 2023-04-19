import sys

from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt


class AddWidget(QWidget):
    sender_signal = pyqtSignal(int)

    def __init__(self, parent=None, widget_id=None, position=None):
        super().__init__(parent=parent)
        self.widget_id = widget_id
        self.setFixedSize(30, 30)
        self.setAcceptDrops(True)
        # self.setDragEnabled(True)
        self.pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")
        self.non_empty_pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")
        self.empty_pixmap = QPixmap("data/resources/icons/Empty.png")
        if position:
            self.move(position)
        self.drag_start_position = self.pos()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        print("paint")
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        painter.end()

    def set_pixmap(self, is_empty=False):
        self.pixmap = self.empty_pixmap if is_empty else self.non_empty_pixmap
        self.update()

    def set_opacity(self, opacity):
        self.setWindowOpacity(opacity)
        self.update()

    def mousePressEvent(self, event):
        print("mouse press")
        if event.button() == Qt.LeftButton:
            # Запоминаем позицию относительно виджета
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        print("mouse move")
        self.sender_signal.emit(self.widget_id)
        if event.buttons() == Qt.LeftButton:
            pixmap = self.grab()
            self.set_pixmap(is_empty=True)
            self.update()
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(self.non_empty_pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)


    def minimumSizeHint(self) -> QtCore.QSize:
        return QSize(30, 30)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSizeHint()


class TestFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        self.widgets = {0: AddWidget(parent=self, widget_id=0, position=QtCore.QPoint(0, 0)),
                        1: AddWidget(parent=self, widget_id=1, position=QtCore.QPoint(50, 50))}

        for widget in self.widgets.values():
            widget.sender_signal.connect(self.set_moved_widget_id)

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

        # self.widget.move(new_point)
        # self.widget.set_pixmap(is_empty=False)
        # self.widget.update()
        widget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())
        widget.move(new_point)
        widget.set_pixmap(is_empty=False)
        widget.update()

        self.moved_widget_id = None
        e.setDropAction(Qt.MoveAction)
        e.accept()

    def set_moved_widget_id(self, widget_id):
        self.moved_widget_id = widget_id



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    frame = TestFrame()
    window2.layout().addWidget(frame)
    window2.show()
    sys.exit(app.exec_())