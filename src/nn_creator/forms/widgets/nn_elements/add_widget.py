import sys

from PyQt5.QtCore import QSize, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt


class AddWidget(QWidget):
    cast_id_signal = pyqtSignal(int)

    def __init__(self, parent=None, widget_id=None, position=None):
        super().__init__(parent=parent)
        self.widget_id = widget_id
        self.WIDGET_SIZE = (30, 30)
        self.setFixedSize(*self.WIDGET_SIZE)
        self.setAcceptDrops(True)
        # self.setDragEnabled(True)
        self.pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png").scaled(*self.WIDGET_SIZE)
        if position:
            self.move(position)
        self.drag_start_position = self.pos()
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        print("paint")
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        painter.end()

    def mousePressEvent(self, event):
        print("mouse press")
        if event.button() == Qt.LeftButton:
            # Запоминаем позицию относительно виджета
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        print("mouse move")

        if event.buttons() == Qt.LeftButton:
            self.cast_id_signal.emit(self.widget_id)
            self.hide()
            self.update()
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(self.pixmap)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        print("leave event")

    def dragLeaveEvent(self, a0: QtGui.QDragLeaveEvent) -> None:
        print("drag leave event")

    # def dropEvent(self, a0: QtGui.QDropEvent) -> None:
    #     print("add widget drop event")

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


class EventFilter(QObject):
    def __init__(self, wigets=None):
        super().__init__()
        self.wigets = wigets if wigets else []

    def eventFilter(self, obj, event):
        # print("Event Filter: sum event happend")
        if event.type() == QEvent.MouseButtonRelease:
            for widget in self.wigets:
                widget.show()
                widget.update()

            print("Event Filter: Mouse Button Release")

        return super().eventFilter(obj, event)

    def update_widgets_list(self, widget):
        if widget not in self.wigets:
            self.wigets.append(widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    frame = TestFrame()
    window2.layout().addWidget(frame)
    window2.show()

    event_filter = EventFilter(list(frame.widgets.values()))
    app.installEventFilter(event_filter)

    sys.exit(app.exec_())
