import sys

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QSize, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QApplication, QMainWindow,
                             QLabel, QFrame, QHBoxLayout, QListWidget,
                             QListWidgetItem)
from nn_creator.forms.widgets.base_classes import BaseNNWidget


class IconLabel(QWidget):
    IconSize = QSize(28, 28)
    HorizontalSpacing = 2

    create_widget_signal = pyqtSignal(BaseNNWidget)

    def __init__(self, icon_pixmap, text, created_widget, parent=None, final_stretch=True):
        super().__init__(parent)
        self.created_widget = created_widget
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setAcceptDrops(True)

        icon = QLabel()
        icon.setPixmap(icon_pixmap.scaled(self.IconSize))

        layout.addWidget(icon)
        layout.addSpacing(self.HorizontalSpacing)
        layout.addWidget(QLabel(text))

        if final_stretch:
            layout.addStretch()

    def mouseMoveEvent(self, event):
        print("mouse move-----")
        # self.sender_signal.emit(self.widget_id)
        if event.buttons() == Qt.LeftButton:
            widget = self.created_widget(parent=self.window())
            widget.hide()
            drag_pixmap = widget.pixmap
            self.create_widget_signal.emit(widget)
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(drag_pixmap)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)


class TestFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(300, 300)
        self.setAcceptDrops(True)

        self.widgets = {}

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

    def update_widgets_holder(self, widget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
        widget.widget_id = key
        widget.setParent(self)
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        # self.moved_widget_id = key


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    window2.setFixedSize(800, 800)

    list_widget = QListWidget(parent=window2)
    list_widget.setFixedSize(300, 300)
    item = QListWidgetItem(list_widget)
    list_widget.addItem(item)

    pixmap = QPixmap("data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png")
    icon_label = IconLabel(icon_pixmap=pixmap, text="add")
    item.setSizeHint(icon_label.minimumSizeHint())
    list_widget.setItemWidget(item, icon_label)
    window2.layout().addWidget(list_widget)

    frame = TestFrame(parent=window2)
    icon_label.create_widget_signal.connect(frame.update_widgets_holder)

    frame.move(0, 300)
    window2.layout().addWidget(frame)

    window2.show()
    sys.exit(app.exec_())
