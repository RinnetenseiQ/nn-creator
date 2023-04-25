import sys

from PyQt5.QtCore import QSize, QMimeData
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QDrag, QMouseEvent
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QGridLayout, QLabel, QFrame, QPushButton, QTreeWidget, \
    QTreeWidgetItem
from PyQt5 import QtCore, QtGui
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import numpy as np


class NNPropertyWidget(QTreeWidget):
    def __init__(self, parent=None, widget_holder=None):
        super().__init__(parent=parent)
        self.widget_holder = widget_holder
        self.setObjectName("property_area")
        # self.setFixedSize(300, 300)
        # self.setAcceptDrops(True)

        self.setHeaderLabels(['Property', "Value"])

        self.widgets = self.widget_holder.widgets
        self.moved_widget_id = None
        self.setStyleSheet("background-color:green;")
        self.update()
        # self.setDragEnabled(True)
        # self.update()

    def display_properties(self, widget_id):
        widget = self.widgets[widget_id]
        self.clear()
        self.addTopLevelItem(QTreeWidgetItem(['widget_id', str(widget_id)]))
        for key, value in widget.cfg['config'].items():
            self.addTopLevelItem(QTreeWidgetItem([str(key), str(value)]))
        self.update()

    def created_widget(self, widget):
        widget.mouse_press_signal.connect(self.display_properties)

    # def set_moved_widget_id(self, widget_id):
    #     print("set_moved_widget_id", widget_id)
    #     self.moved_widget_id = widget_id

    # def update_widgets_holder(self, widget):
    #     key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 0
    #     # widget.widget_id = key
    #     # widget.setParent(self)
    #     # widget.cast_id_signal.connect(self.set_moved_widget_id)
    #     # widget.delete_widget_signal.connect(self.delete_widget_id)
    #     widget.mouse_press_signal.connect(self.display_properties)
    #     self.widgets[key] = widget
    #     # self.set_moved_widget_id(key)
    #     # self.moved_widget_id = key
    #     print(f"prop_widgets_ids: {self.widgets.keys()}")