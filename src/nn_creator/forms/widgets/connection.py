import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QPaintEvent, QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QFrame


# from nn_creator.forms.widgets.base_classes import BaseNNWidget


def get_angle(x1, y1, x2, y2):
    a = y2 - y1
    c = x2 - x1
    b = np.sqrt(a ** 2 + c ** 2)

    if a == 0 and b == c:
        angle = 0
    elif c == 0 and -a == b:
        angle = 90
    elif a == 0 and b == -c:
        angle = 180
    elif c == 0 and a == b:
        angle = 270
    elif a < 0 and b > 0:
        angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2.0 * b * c)))
    else:
        angle = 360 - np.degrees(np.arccos((b * b + c * c - a * a) / (2.0 * b * c)))

    return angle


def get_coordinates(base_vector_start, base_vector_end, angle_degree, vector_len):
    # задаем начальные координаты неизвестного вектора

    angle = angle_degree * np.pi / 180

    x1, y1 = base_vector_start.x(), base_vector_start.y()
    x2, y2 = base_vector_end.x(), base_vector_end.y()

    start_point = (x2, y2)

    dx = x2 - x1
    dy = y2 - y1
    base_vector_length = np.sqrt(dx ** 2 + dy ** 2)
    direction_vector = (dx / base_vector_length, dy / base_vector_length)

    # вычисляем координаты конца вектора
    end_point_x = start_point[0] + vector_len * np.cos(angle) * direction_vector[0] - vector_len * np.sin(angle) * \
                  direction_vector[1]
    end_point_y = start_point[1] + vector_len * np.cos(angle) * direction_vector[1] + vector_len * np.sin(angle) * \
                  direction_vector[0]
    end_point = (end_point_x, end_point_y)

    return start_point, end_point


class ConnectionEventFilter(QObject):

    def __init__(self):
        super().__init__()


class ConnectionWidget(QWidget):
    delete_signal = pyqtSignal(int)

    def __init__(self,
                 # start_widget: BaseNNWidget,
                 start_widget,
                 connection_id=None, parent=None):
        super().__init__(parent=parent)
        self.setParent(parent)
        self.start_widget = start_widget
        self.end_widget = None
        self.connection_id = connection_id if connection_id else None
        self.paint_mode = True

        widget_size = start_widget.geometry()
        widget_x = widget_size.x()
        widget_y = widget_size.y()
        start_x = widget_x + widget_size.width() / 2
        start_y = widget_y + widget_size.height()
        self.start_position = QPoint(start_x, start_y)
        self.end_position: QPoint = self.cursor().pos()

        self.move(start_widget.pos())  # ???
        a = self.pos()
        b = self.start_widget.pos()
        d = self.start_widget.geometry()
        c = self.geometry()
        self.update()
        # self.repaint()

    def setParent(self, parent: 'QWidget') -> None:
        super(ConnectionWidget, self).setParent(parent)
        self.setGeometry(parent.geometry())
        # self.move(parent.pos())

    def paintEvent(self, event: QPaintEvent) -> None:
        print("connection paint event")
        b = self.start_widget.pos()
        a = self.pos()
        c = self.start_widget.geometry()
        if self.paint_mode:
            painter = QPainter(self.parent())
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QtCore.Qt.red)
            painter.setBrush(QtCore.Qt.white)

            # angle = get_angle(x1, y1, x2, y2)
            x1, y1 = self.start_position.x(), self.start_position.y()
            x2, y2 = self.end_position.x(), self.end_position.y()
            painter.drawLine(x1, y1, x2, y2)
            (x1, y1), (x2, y2) = get_coordinates(self.start_position, self.end_position, 135, 10)
            painter.drawLine(x1, y1, x2, y2)
            (x1, y1), (x2, y2) = get_coordinates(self.start_position, self.end_position, -135, 10)
            painter.drawLine(x1, y1, x2, y2)
            painter.end()

    def mouseMoveEvent(self, event) -> None:
        print("connection move event")
        position = event.pos()
        self.end_position = position
        self.update()

    def leaveEvent(self, event) -> None:
        print("connection leave event")

    def set_end_widget(self,
                       # widget: BaseNNWidget,
                       widget
                       ):
        self.end_widget = widget
        widget_size = widget.geometry()
        widget_x = widget_size.x()
        widget_y = widget_size.y()
        start_x = widget_x + widget_size.width() / 2
        start_y = widget_y
        self.end_position = QPoint(start_x, start_y)

    def set_paint_mode(self, flag: bool):
        self.paint_mode = flag


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setFixedSize(1800, 1000)
    frame = QFrame(parent=window)
    frame.setStyleSheet("background-color:yellow;")
    frame.setFixedSize(window.size())
    button = QPushButton(parent=frame, text="pizdec")
    button.setFixedSize(100, 30)
    button.setStyleSheet("background-color:blue;")
    widget = ConnectionWidget(start_widget=button, parent=frame)
    # widget.setStyleSheet("background-color:blue;")
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())
