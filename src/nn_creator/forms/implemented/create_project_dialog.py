from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys
import os

from nn_creator.forms.from_ui.CreateProjectDialog_parent import Ui_CreateProjectDialog
from nn_creator.forms.implemented.project_editor_window import ProjectEditorWindow


class CreateProjectDialog(QtWidgets.QDialog, Ui_CreateProjectDialog):
    def __init__(self, parent,  project_windows):
        super().__init__()
        self._parent = parent
        self.project_windows = project_windows
        self.setupUi(self)
        self._init_widgets()
        self._connect_all()

    def _init_widgets(self):
        self.project_name_LE.setText("project")

    def _connect_all(self):
        self.create_project_btn.clicked.connect(self._create_project_btn_clicked)
        self.open_dir_btn.clicked.connect(self._open_dir_btn_clicked)
        self.cancel_btn.clicked.connect(self._cancel_btn_clicked)

        self.project_name_LE.textEdited.connect(self._project_name_LE_edited)
        self.project_path_LE.textEdited.connect(self._project_path_LE_edited)

    def _project_name_LE_edited(self):
        text = self.project_path_LE.text()
        location = "/".join(text.split("/")[:-1])
        name = self.project_name_LE.text()
        project_path = "{}/{}".format(location, name)
        self.project_path_LE.setText(project_path)

    def _project_path_LE_edited(self):
        text = self.project_path_LE.text()
        project_name = text.split("/")[-1]
        self.project_name_LE.setText(project_name)


    def _create_project_btn_clicked(self):
        project_path = self.project_path_LE.text()
        os.makedirs(project_path, exist_ok=True)
        project_window = ProjectEditorWindow(project_path)
        self.project_windows.append(project_window)
        project_window.show()
        self._parent.hide()
        self.close()

    def _open_dir_btn_clicked(self):
        dirlist = QFileDialog.getExistingDirectory(parent=self,
                                                   caption="Open project",
                                                   # directory=self.settings.value("projectPath", "C:/",type=str)
                                                   )
        text = "{}/{}".format(dirlist, self.project_name_LE.text())
        self.project_path_LE.setText(text)

    def _cancel_btn_clicked(self):
        self.close()
