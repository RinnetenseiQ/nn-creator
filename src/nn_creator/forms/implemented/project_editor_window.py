from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMainWindow
import sys

from nn_creator.forms.from_ui.ProjectWindow_parent import Ui_ProjectEditorWindow


class ProjectEditorWindow(QMainWindow, Ui_ProjectEditorWindow):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.setupUi(self)
        self._init_widgets()
        self._connect_all()

    def _init_widgets(self):
        pass

    def _connect_all(self):
        pass
