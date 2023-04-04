from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMainWindow
import sys
import pandas as pd

from nn_creator.forms.from_ui.ProjectWindow_parent import Ui_ProjectEditorWindow
from nn_creator.forms.widgets.pandas_model import PandasModel



class ProjectEditorWindow(QMainWindow, Ui_ProjectEditorWindow):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.setupUi(self)
        self._init_widgets()
        self._connect_all()

    def _init_widgets(self):
        self.dataset_type_CB.addItems(["Table(1D, .csv, .txt, .xlsx, etc)",
                                       "Labeled images"])

    def _connect_all(self):
        self.get_dataset_dir_btn.clicked.connect(self._get_dataset_dir_btn_clicked)
        self.open_dataset_btn.clicked.connect(self._open_dataset_btn_clicked)

        # self.dataset_list_LW.currentItemChanged


    def _open_dataset_btn_clicked(self):
        path = self.dataset_path_LE.text()
        if self.dataset_type_CB.currentIndex() in [0]:
            data = pd.read_csv(path, sep='\t').reset_index(drop=True).dropna(axis=1)
            model = PandasModel(data)
            self.dataframe_TV.setModel(model)
        elif self.dataset_type_CB.currentIndex() in [1]:
            pass

    def _get_dataset_dir_btn_clicked(self):
        if self.dataset_type_CB.currentIndex() in [0]:
            path = QFileDialog.getOpenFileName(parent=self, caption="Open Table-like dataset file")
        elif self.dataset_type_CB.currentIndex() in [1]:
            path = QFileDialog.getExistingDirectory(parent=self,
                                                    caption="Open Labeled images dataset folder",
                                                    # directory=self.settings.value("projectPath", "C:/",type=str)
                                                    )
        else:
            raise NotImplementedError()

        self.dataset_path_LE.setText(path[0])
