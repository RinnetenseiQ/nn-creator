import sys

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame


# class CursorPos(QObject):
#     def __init__(self, window):
#         super().__init__()
#         self.window = window
#
#     def run(self):
#         pos = self.window.mapFromGlobal(self.cursor().pos())
#         x = pos.x()
#         y = pos.y()
#         print(f"Cursor position: x={x}, y={y}")


class NNSchemeWidget(QFrame):
    set_moved_widget_id_signal = pyqtSignal(int)

    #def __init__(self, parent=None, event_filter: GlobalEventFilter2 = None):
    def __init__(self, parent=None, widget_holder=None):
        super().__init__(parent=parent)
        self.widget_holder = widget_holder
        self.setObjectName("yelow")
        # self._event_filter = event_filter
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)
        self.widgets = self.widget_holder.widgets

        self.moved_widget_id = None
        self.setStyleSheet("background-color:yellow;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    # @property
    # def event_filter(self):
    #     return self._event_filter
    #
    # @event_filter.setter
    # def event_filter(self, value: GlobalEventFilter2):
    #     self._event_filter = value

    def dragEnterEvent(self, e):
        print(f"drag id: {self.moved_widget_id}")
        # self.updateCursorPosition()
        # self.drag_widget = self.sender()
        e.accept()

    def dropEvent(self, e):
        position = e.pos()
        # if self.event_filter.nn_scheme_moved_widget_id < 0: return
        # widget = self.event_filter.nn_scheme_widgets[self.event_filter.nn_scheme_moved_widget_id]

        widget = self.widgets[self.moved_widget_id]
        new_point = QtCore.QPoint(position.x() - widget.drag_start_position.x(),
                                  position.y() - widget.drag_start_position.y())

        widget.move(new_point)
        widget.show()
        widget.update()
        # self.set_moved_widget_id_signal.emit(-1)

        print(f"drop id: {self.moved_widget_id}")
        self.moved_widget_id = None
        e.setDropAction(Qt.MoveAction)
        e.accept()

        # @pyqtSlot()
        # def update_children(self, child: QWidget):
        #     child.setParent(self)
        #     if isinstance(child, ConnectionWidget):
        #         child.set_paint_mode(True)
        #         print(child.geometry())
        #         print(self.geometry())
        #         child.show()


    def created_widget(self, widget):
        widget.setParent(self)
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        self.set_moved_widget_id(widget.widget_id)

    def set_moved_widget_id(self, widget_id):
        print("set_moved_widget_id", widget_id)
        self.moved_widget_id = widget_id



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window2 = QMainWindow()
    window2.layout().addWidget(NNSchemeWidget())
    window2.show()
    sys.exit(app.exec_())
