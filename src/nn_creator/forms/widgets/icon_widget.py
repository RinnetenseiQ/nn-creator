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

    def __init__(self, icon_pixmap, text, created_widget, parent=None, final_stretch=True, event_filter=None):
        super().__init__(parent)
        self.event_filter = event_filter
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
            widget: BaseNNWidget = self.created_widget(parent=self.window(), event_filter=self.event_filter)
            widget.hide()
            drag_pixmap = widget.pixmap
            self.create_widget_signal.emit(widget)
            widget.connect_signal.connect(self.event_filter.add_end_connection_widget)
            mime_data = QtCore.QMimeData()
            drag = QtGui.QDrag(self)
            drag.setMimeData(mime_data)
            drag.setPixmap(drag_pixmap)
            drag.setHotSpot(event.pos() - self.rect().topLeft())
            drag.exec_(Qt.CopyAction | Qt.MoveAction)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window2 = QMainWindow()
    window2.setFixedSize(800, 800)

    # window2.layout().addWidget(frame)

    window2.show()
    sys.exit(app.exec_())
