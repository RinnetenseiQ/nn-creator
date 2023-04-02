from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from nn_creator.forms.implemented.start_window import NNCreatorStartWindow
import sys

if __name__ == '__main__':
    # Новый экземпляр QApplication
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = NNCreatorStartWindow()
    mainWindow.show()
    # Запуск
    sys.exit(app.exec_())