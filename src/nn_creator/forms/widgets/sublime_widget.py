import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.Qsci import QsciScintilla, QsciLexerPython

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        scintilla_edit = QsciScintilla()
        scintilla_edit.setLexer(QsciLexerPython())
        vbox.addWidget(scintilla_edit)
        self.setLayout(vbox)
        self.setGeometry(100, 100, 800, 600)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    sys.exit(app.exec_())