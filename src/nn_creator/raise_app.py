from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from nn_creator.forms.implemented.start_window import NNCreatorStartWindow
import sys

from nn_creator.forms.widgets.nn_elements.add_widget import EventFilter

if __name__ == '__main__':
    # Новый экземпляр QApplication
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = NNCreatorStartWindow()
    mainWindow.show()

    event_filter = EventFilter()
    app.installEventFilter(event_filter)
    # Запуск
    sys.exit(app.exec_())