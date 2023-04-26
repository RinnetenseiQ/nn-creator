import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QPoint, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPaintEvent, QPainter, QCursor
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QFrame
from typing import Union


# from nn_creator.forms.widgets.base_classes import BaseNNWidget

class Magnet(QThread):
    update_connection_signal = pyqtSignal(list)

    def __init__(self, start_widget: Union[QWidget, QCursor], end_widget: Union[QWidget, QCursor]):
        super().__init__()
        self.end_widget = end_widget
        self.start_widget = start_widget
        self.is_break = False

    def run(self) -> None:
        while True:

            if isinstance(self.start_widget, QCursor):
                pos = self.start_widget.pos()
                start_x = pos.x()
                start_y = pos.y()
            else:
                start_x = self.start_widget.x() + self.start_widget.width() / 2
                start_y = self.start_widget.y() + self.start_widget.height()

            if isinstance(self.end_widget, QCursor):
                pos = self.end_widget.pos()
                end_x = pos.x()
                end_y = pos.y()
            else:
                end_x = self.end_widget.x() + self.end_widget.width() / 2
                end_y = self.end_widget.y()

            self.update_connection_signal.emit([start_x, start_y, end_x, end_y])
            if self.is_break: break
            self.msleep(1)
            # self.sleep(0.01)

    def set_start_widget(self, widget: list):
        self.start_widget = widget[0]

    def set_end_widget(self, widget: list):
        self.end_widget = widget[0]

    def cancel(self):
        self.is_break = True


def get_coordinates(base_vector_start, base_vector_end, angle_degree, vector_len):
    angle = angle_degree * np.pi / 180

    x1, y1 = base_vector_start.x(), base_vector_start.y()
    x2, y2 = base_vector_end.x(), base_vector_end.y()

    start_point = (x2, y2)

    dx = x2 - x1
    dy = y2 - y1
    base_vector_length = np.sqrt(dx ** 2 + dy ** 2)
    direction_vector = (dx / base_vector_length, dy / base_vector_length)

    end_point_x = start_point[0] + vector_len * np.cos(angle) * direction_vector[0] - vector_len * np.sin(angle) * \
                  direction_vector[1]
    end_point_y = start_point[1] + vector_len * np.cos(angle) * direction_vector[1] + vector_len * np.sin(angle) * \
                  direction_vector[0]
    end_point = (round(end_point_x), round(end_point_y))

    return start_point, end_point


class ConnectionWidget(QWidget):
    delete_signal = pyqtSignal(int)
    cancelation_signal = pyqtSignal()
    set_start_widget_signal = pyqtSignal(list)
    set_end_widget_signal = pyqtSignal(list)

    def __init__(self,
                 # start_widget: BaseNNWidget,
                 start_widget: QWidget = None,
                 end_widget: QWidget = None,
                 connection_id=None,
                 parent=None):
        super().__init__(parent=parent)
        assert start_widget or end_widget

        self.setParent(parent)
        self.start_widget = start_widget if start_widget else self.cursor()
        self.end_widget = end_widget if end_widget else self.cursor()

        self.connection_id = connection_id
        self.paint_mode = True

        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

        self.thread = Magnet(start_widget=self.start_widget, end_widget=self.end_widget)
        self.thread.update_connection_signal.connect(self.update_position)
        self.cancelation_signal.connect(self.thread.cancel)
        self.set_start_widget_signal.connect(self.thread.set_start_widget)
        self.set_end_widget_signal.connect(self.thread.set_end_widget)

        self.thread.start()
        self.update()

    def setParent(self, parent: 'QWidget') -> None:
        super(ConnectionWidget, self).setParent(parent)
        self.setGeometry(parent.geometry())
        # self.move(parent.pos())

    def paintEvent(self, event: QPaintEvent) -> None:
        print("connection paint event")
        if self.paint_mode:
            painter = QPainter(self.parent())
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QtCore.Qt.red)
            painter.setBrush(QtCore.Qt.white)

            # angle = get_angle(x1, y1, x2, y2)
            x1, y1 = self.start_x, self.start_y
            x2, y2 = self.end_x, self.end_y
            painter.drawLine(x1, y1, x2, y2)
            (x1, y1), (x2, y2) = get_coordinates(QPoint(self.start_x, self.start_y),
                                                 QPoint(self.end_x, self.end_y), 150, 10)
            painter.drawLine(x1, y1, x2, y2)
            (x1, y1), (x2, y2) = get_coordinates(QPoint(self.start_x, self.start_y),
                                                 QPoint(self.end_x, self.end_y), -150, 10)
            painter.drawLine(x1, y1, x2, y2)
            painter.end()

    # def mouseMoveEvent(self, event) -> None:
    #     print("connection move event")
    #     position = event.pos()
    #     self.end_position = position
    #     self.update()

    def leaveEvent(self, event) -> None:
        print("connection leave event")

    def set_start_widget(self, widget ):
        self.set_end_widget_signal.emit([widget])

    def set_end_widget(self, widget):
        self.set_end_widget_signal.emit([widget])
        # self.end_widget = widget
        # widget_size = widget.geometry()
        # widget_x = widget_size.x()
        # widget_y = widget_size.y()
        # start_x = widget_x + widget_size.width() / 2
        # start_y = widget_y
        # self.end_position = QPoint(start_x, start_y)

    def set_paint_mode(self, flag: bool):
        self.paint_mode = flag

    def update_position(self, coordinates: list):
        self.start_x, self.start_y, self.end_x, self.end_y = coordinates
        self.update()



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

    button2 = QPushButton(parent=frame, text="pizdec2")
    button2.setFixedSize(100, 30)
    button2.setStyleSheet("background-color:blue;")
    button2.move(300, 300)

    widget = ConnectionWidget(start_widget=button, parent=frame)
    # widget.setStyleSheet("background-color:blue;")
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())
