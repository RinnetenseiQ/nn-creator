import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFrame, QPushButton
from PyQt5.QtGui import QPainter, QPen, QCursor, QPainterPath
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect
import numpy as np


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
    delete_connection_signal = pyqtSignal(int)

    def __init__(self, start_widget: QWidget = None, end_widget: QWidget = None, parent=None, connection_id=None, event_filter=None):
        super().__init__(parent)
        # self.current_pos = None
        self.event_filter = event_filter
        self.end_widget = end_widget
        self.start_widget = start_widget
        if self.parent():
            self.setGeometry(self.parent().geometry())
        self.connection_id = connection_id
        self.is_end_moved = True

        self.start_x = start_widget.x() + start_widget.width() / 2 if start_widget else 0
        self.start_y = start_widget.y() + start_widget.height() if start_widget else 0

        self.mid_xs = []
        self.mid_ys = []

        self.end_x = end_widget.x() + end_widget.width() / 2 if end_widget else 0
        self.end_y = end_widget.y() if end_widget else 0

        self.shift_length = 20

        self.is_paint_mode = False if start_widget and end_widget else True
        self.setMouseTracking(self.is_paint_mode)
        # self.update()

    def setParent(self, parent: QWidget) -> None:
        super(ConnectionWidget, self).setParent(parent)
        self.setGeometry(parent.geometry())

    def set_moved(self, flag: bool):
        self.is_end_moved = flag

    def set_paint_mode(self, flag: bool):
        self.is_paint_mode = flag
        self.setMouseTracking(flag)

    def set_start_point(self, point: QPoint):
        self.start_x = point.x()
        self.start_y = point.y()

    # def update(self) -> None:
    #     x = self.cursor().pos().x()
    #     y = self.cursor().pos().y()
    #
    #     self.start_x = self.start_widget.x() + self.start_widget.width() / 2 if self.start_widget else x
    #     self.start_y = self.start_widget.y() + self.start_widget.height() if self.start_widget else y
    #
    #     self.end_x = self.end_widget.x() + self.end_widget.width() / 2 if self.end_widget else x
    #     self.end_y = self.end_widget.y() if self.end_widget else y
    #
    #     self.calculate_mid_points()
    #
    #     super(ConnectionWidget, self).update()

    def set_start_widget(self, widget):
        self.start_widget = widget
        self.start_x = self.start_widget.x() + self.start_widget.width() / 2
        self.start_y = self.start_widget.y() + self.start_widget.height()
        self.calculate_mid_points()
        self.update()

    def set_end_widget(self, widget):
        self.end_widget = widget
        self.end_x = self.end_widget.x() + self.end_widget.width() / 2
        self.end_y = self.end_widget.y()
        self.calculate_mid_points()
        self.update()

    def calculate_mid_points(self):

        if self.start_x > self.end_x and self.start_y > self.end_y:
            mid_x = (self.start_x + self.end_x) / 2
            self.mid_xs = [mid_x, mid_x]
            self.mid_ys = [self.start_y + self.shift_length, self.end_y - self.shift_length]
        elif self.start_x > self.end_x and self.start_y < self.end_y:
            mid_y = (self.start_y + self.end_y) / 2
            self.mid_xs = [self.start_x, self.end_x]
            self.mid_ys = [mid_y, mid_y]
        elif self.start_x < self.end_x and self.start_y > self.end_y:
            mid_x = (self.start_x + self.end_x) / 2
            self.mid_xs = [mid_x, mid_x]
            self.mid_ys = [self.start_y + self.shift_length, self.end_y - self.shift_length]
        elif self.start_x < self.end_x and self.start_y < self.end_y:
            mid_y = (self.start_y + self.end_y) / 2
            self.mid_xs = [self.start_x, self.end_x]
            self.mid_ys = [mid_y, mid_y]

    def set_end_point(self, point: QPoint):
        self.end_x = point.x()
        self.end_y = point.y()

    def update_widgets(self):
        self.start_x = self.start_widget.x() + self.start_widget.width() / 2
        self.start_y = self.start_widget.y() + self.start_widget.height()

        self.end_x = self.end_widget.x() + self.end_widget.width() / 2
        self.end_y = self.end_widget.y()
        self.calculate_mid_points()
        self.is_paint_mode = True
        self.update()
        self.is_paint_mode = False

    def paintEvent(self, event):
        # if self.is_paint_mode:
        painter = QPainter()
        painter.begin(self)

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))

        path = QPainterPath()
        self.calculate_mid_points()
        path.moveTo(self.start_x, self.start_y)
        path.lineTo(self.start_x, self.start_y + self.shift_length)

        for x, y in zip(self.mid_xs, self.mid_ys):
            path.lineTo(x, y)

        path.lineTo(self.end_x, self.end_y - self.shift_length)
        path.lineTo(self.end_x, self.end_y)

        painter.drawPath(path)

        (x1, y1), (x2, y2) = get_coordinates(QPoint(self.end_x, self.end_y - self.shift_length),
                                             QPoint(self.end_x, self.end_y), 150, 10)
        painter.drawLine(x1, y1, x2, y2)
        (x1, y1), (x2, y2) = get_coordinates(QPoint(self.end_x, self.end_y - self.shift_length),
                                             QPoint(self.end_x, self.end_y), -150, 10)
        painter.drawLine(x1, y1, x2, y2)
        painter.end()
        # painter = QPainter(self)
        #
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        # if self.current_pos:  # проверяем, что атрибут задан
        #     painter.drawLine(self.width() // 2, self.height() // 2,
        #                      self.current_pos.x(), self.current_pos.y())

    def mouseMoveEvent(self, event):
        # print("connection move event")
        if self.is_paint_mode:
            if self.is_end_moved:
                self.end_x = event.pos().x()
                self.end_y = event.pos().y()
            else:
                self.start_x = event.pos().x()
                self.start_y = event.pos().y()
            self.update()

    def mousePressEvent(self, event) -> None:
        if self.is_paint_mode:
            pos = event.pos()
            for widget in self.event_filter.nn_scheme_widgets.values():
                geom: QRect = widget.geometry()
                if geom.contains(pos):
                    self.set_paint_mode(False)
                    self.set_end_widget(widget)
                    widget.input_connections.append(self)

                widget.raise_()
        print("connection press event")

    # def


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

    widget = ConnectionWidget(start_widget=button,
                              # end_widget=button2,
                              parent=frame)
    # widget.setStyleSheet("background-color:blue;")
    window.layout().addWidget(frame)
    window.show()

    sys.exit(app.exec_())

#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     widget = ConnectionWidget()
#     widget.resize(400, 400)
#     widget.show()
#     sys.exit(app.exec_())
