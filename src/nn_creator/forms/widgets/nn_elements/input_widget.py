from PyQt5.QtGui import QPaintEvent
from PyQt5.QtWidgets import QWidget


class InputLayerWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        pass

