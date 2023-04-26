import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QTreeWidgetItem, QTreeWidget
from nn_creator.forms.from_ui.ProjectWindow_parent import Ui_ProjectEditorWindow
from nn_creator.forms.utils.event_filters import GlobalEventFilter2
from nn_creator.forms.widgets.icon_widget import IconLabel
from nn_creator.forms.widgets.nn_property import NNPropertyWidget
from nn_creator.forms.widgets.nn_scheme import NNSchemeWidget
from nn_creator.forms.widgets.pandas_model import PandasModel
import nn_creator.forms.widgets.nn_elements as nn_widgets
from nn_creator.forms.widgets.base_classes import BaseNNWidget

non_trainable_widgets = [nn_widgets.InputWidget,
                         nn_widgets.AddWidget,
                         nn_widgets.ActivationWidget]
non_trainable_widgets_labels = ["Input", "Add", "Activation"]
non_trainable_widgets_icons = [
    "data/resources/icons/Example_Theme/layers/input_layer/icons8-несколько-входов-24.png",
    "data/resources/icons/Example_Theme/layers/add/icons8-добавить-50.png",
    "data/resources/icons/Example_Theme/layers/activation/icons8-curve-64.png",
]

trainable_widgets = [nn_widgets.DenseWidget]
trainable_widgets_labels = ["Dense"]
trainable_widgets_icons = [
    "data/resources/icons/Example_Theme/layers/dense/icons8-mean-32.png",
]


class ProjectEditorWindow(QMainWindow, Ui_ProjectEditorWindow):
    update_event_filter_signal = pyqtSignal(BaseNNWidget)

    def __init__(self, path, event_filter: GlobalEventFilter2 = None):
        super().__init__()
        self._event_filter = event_filter
        self.path = path
        self.setupUi(self)
        self._init_widgets()
        self._connect_all()

    @property
    def event_filter(self):
        return self._event_filter

    def _init_widgets(self):
        self.dataset_type_CB.addItems(["Table(1D, .csv, .txt, .xlsx, etc)",
                                       "Labeled images"])


        b = NNPropertyWidget(parent=self.model_properties_TW.parent(), event_filter=self.event_filter)
        b.setFixedSize(self.model_properties_TW.size())
        self.model_properties_TW = b

        a = NNSchemeWidget(parent=self.scrollArea.parent(), event_filter=self.event_filter)
        a.setFixedSize(self.scrollArea.size())
        self.scrollArea = a

        self.event_filter.connection_create_signal.connect(self.scrollArea.update_children)
        trainable_group_item = self.model_blocks_TW.topLevelItem(0)
        trainable_group_item.setExpanded(True)
        non_trainable_group_item = self.model_blocks_TW.topLevelItem(1)
        non_trainable_group_item.setExpanded(True)

        for widget, label, icon_path in zip(non_trainable_widgets,
                                            non_trainable_widgets_labels,
                                            non_trainable_widgets_icons):
            temp = QTreeWidgetItem(non_trainable_group_item)
            non_trainable_group_item.addChild(temp)
            icon_widget = IconLabel(icon_pixmap=QPixmap(icon_path), text=label, created_widget=widget, event_filter=self.event_filter)
            self.model_blocks_TW.setItemWidget(temp, 0, icon_widget)
            icon_widget.create_widget_signal.connect(self.scrollArea.update_children)
            icon_widget.create_widget_signal.connect(self.event_filter.update_nn_scheme_widgets_list)

        for widget, label, icon_path in zip(trainable_widgets,
                                            trainable_widgets_labels,
                                            trainable_widgets_icons):
            temp = QTreeWidgetItem(trainable_group_item)
            trainable_group_item.addChild(temp)
            icon_widget = IconLabel(icon_pixmap=QPixmap(icon_path), text=label, created_widget=widget, event_filter=self.event_filter)
            self.model_blocks_TW.setItemWidget(temp, 0, icon_widget)
            icon_widget.create_widget_signal.connect(self.scrollArea.update_children)
            icon_widget.create_widget_signal.connect(self.event_filter.update_nn_scheme_widgets_list)

        print("")

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
