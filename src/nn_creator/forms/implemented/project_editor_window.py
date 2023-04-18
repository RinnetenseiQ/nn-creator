from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMainWindow
import sys
import pandas as pd
import math
import fileinput
from nn_creator.forms.from_ui.ProjectWindow_parent import Ui_ProjectEditorWindow
from nn_creator.forms.widgets.pandas_model import PandasModel
from PyQt5.QtCore import QThread


class DataFrameLoader(QThread):
    def __init__(self):
        super().__init__()

    def run(self) -> None:
        pass

class ProjectEditorWindow(QMainWindow, Ui_ProjectEditorWindow):
    #TODO: добавить поле ввода сепаратора
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.setupUi(self)
        self._init_widgets()
        self._connect_all()
        self.ROWS_IN_TABLE = 6
        self.SEP = ","

    def _init_widgets(self):
        self.dataset_type_CB.addItems(["Table(1D, .csv, .txt, .xlsx, etc)",
                                       "Labeled images"])

    def _connect_all(self):
        self.get_dataset_dir_btn.clicked.connect(self._get_dataset_dir_btn_clicked)
        self.open_dataset_btn.clicked.connect(self._open_dataset_btn_clicked)
        self.table_scroll_VSB.valueChanged.connect(self._change_slider_pos)
        # self.dataset_list_LW.currentItemChanged

    def _open_dataset_btn_clicked(self):
        path = self.dataset_path_LE.text()
        if self.dataset_type_CB.currentIndex() in [0]:
            self.num_data_lines, self.max_slider_pos = self._line_counter(path, rows_in_table=self.ROWS_IN_TABLE)
            data = self._df_sample_for_table(filepath=path,
                                             sep=self.SEP,
                                             slider_pos=0,
                                             num_data_lines=self.num_data_lines,
                                             rows_in_table=self.ROWS_IN_TABLE)

            # data = pd.read_csv(path, sep='\t').reset_index(drop=True).dropna(axis=1)
            self.table_scroll_VSB.setMaximum(self.max_slider_pos)
            model_for_table = PandasModel(data)
            self.dataframe_TV.setModel(model_for_table)
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

    def _change_slider_pos(self):
        path = self.dataset_path_LE.text()
        pos = self.table_scroll_VSB.sliderPosition()

        data = self._df_sample_for_table(filepath=path,
                                         sep=self.SEP,
                                         slider_pos=pos,
                                         num_data_lines=self.num_data_lines,
                                         rows_in_table=self.ROWS_IN_TABLE)

        model_for_table = PandasModel(data)
        self.dataframe_TV.setModel(model_for_table)

    def _line_counter(self, filepath: str, rows_in_table: int, is_header=True):
        num_data_lines = 0
        with fileinput.input(files=filepath) as f:
            for line in f:
                num_data_lines += 1
        if is_header:
            num_data_lines = num_data_lines - 1
        max_slider_pos = math.ceil(num_data_lines / rows_in_table)
        return num_data_lines, max_slider_pos

    def _df_sample_for_table(self, filepath: str, rows_in_table: int, header: list = None, num_data_lines=1, slider_pos=1, sep=","):
        # возможно бесполезная хрень
        if not header:
            header = pd.read_csv(filepath, skipfooter=num_data_lines, engine='python').columns.tolist()

        skiphead = slider_pos * rows_in_table
        # TODO: add check skipfoot
        skipfoot = num_data_lines - skiphead - rows_in_table
        if skipfoot < 0:
            skipfoot = 0
        df = pd.read_csv(filepath, sep=sep, skiprows=skiphead, skipfooter=skipfoot, engine='python')
        df.columns = header
        df = df.set_index(pd.Index(i for i in range(skiphead, skiphead + rows_in_table)))
        return df

