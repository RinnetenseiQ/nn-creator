from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QFileDialog
import sys

from nn_creator.forms.implemented.create_project_dialog import CreateProjectDialog
from nn_creator.forms.from_ui.StartWindow_parent import Ui_NNCreatorStartWindow
from nn_creator.forms.implemented.project_editor_window import ProjectEditorWindow
from nn_creator.forms.widgets.nn_elements.add_widget import EventFilter


class NNCreatorStartWindow(QtWidgets.QMainWindow, Ui_NNCreatorStartWindow):
    def __init__(self, event_filter=None):
        super().__init__()
        self.event_filter = event_filter
        self.setupUi(self)
        self.project_windows = []
        self._init_widgets()
        self._connect_all()

    def _init_widgets(self):
        pass

    def _connect_all(self):
        self.new_project_btn.clicked.connect(self._new_project_btn_Clicked)
        self.open_project_btn.clicked.connect(self._open_project_btn_Clicked)

    def _new_project_btn_Clicked(self):
        self.create_project_window = CreateProjectDialog(parent=self,
                                                         project_windows=self.project_windows,
                                                         event_filter=self.event_filter)
        self.create_project_window.exec()

    def _open_project_btn_Clicked(self):
        path = QFileDialog.getExistingDirectory(parent=self,
                                                caption="Open project",
                                                # directory=self.settings.value("projectPath", "C:/",type=str)
                                                )
        project_window = ProjectEditorWindow(path=path, event_filter=self.event_filter)
        self.project_windows.append(project_window)
        project_window.show()
        self.hide()


if __name__ == '__main__':
    QCoreApplication.setOrganizationName("QSoft")
    QCoreApplication.setApplicationName("NN Family Creater")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    event_filter = EventFilter()
    app.installEventFilter(event_filter)

    capture_window = NNCreatorStartWindow(event_filter=event_filter)
    capture_window.show()
    # Запуск
    sys.exit(app.exec_())
