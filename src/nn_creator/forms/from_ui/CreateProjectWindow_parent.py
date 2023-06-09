# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/nn_creator/forms/ui/CreateProjectWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CreateProjectWindow(object):
    def setupUi(self, CreateProjectWindow):
        CreateProjectWindow.setObjectName("CreateProjectWindow")
        CreateProjectWindow.resize(832, 218)
        self.centralwidget = QtWidgets.QWidget(CreateProjectWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 791, 101))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.project_name_LE = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.project_name_LE.setObjectName("project_name_LE")
        self.verticalLayout_2.addWidget(self.project_name_LE)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 1, -1, 1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.project_path_LE = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.project_path_LE.setObjectName("project_path_LE")
        self.horizontalLayout_4.addWidget(self.project_path_LE)
        self.open_dir_btn = QtWidgets.QToolButton(self.horizontalLayoutWidget_2)
        self.open_dir_btn.setObjectName("open_dir_btn")
        self.horizontalLayout_4.addWidget(self.open_dir_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.task_CB = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.task_CB.setObjectName("task_CB")
        self.verticalLayout_2.addWidget(self.task_CB)
        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 20)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(600, 120, 195, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cancel_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.cancel_btn.setObjectName("cancel_btn")
        self.horizontalLayout_2.addWidget(self.cancel_btn)
        self.create_project_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.create_project_btn.setObjectName("create_project_btn")
        self.horizontalLayout_2.addWidget(self.create_project_btn)
        CreateProjectWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(CreateProjectWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 832, 25))
        self.menubar.setObjectName("menubar")
        CreateProjectWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(CreateProjectWindow)
        self.statusbar.setObjectName("statusbar")
        CreateProjectWindow.setStatusBar(self.statusbar)

        self.retranslateUi(CreateProjectWindow)
        QtCore.QMetaObject.connectSlotsByName(CreateProjectWindow)

    def retranslateUi(self, CreateProjectWindow):
        _translate = QtCore.QCoreApplication.translate
        CreateProjectWindow.setWindowTitle(_translate("CreateProjectWindow", "NN-creator Create project"))
        self.label_3.setText(_translate("CreateProjectWindow", "Project Name"))
        self.label.setText(_translate("CreateProjectWindow", "Location"))
        self.label_2.setText(_translate("CreateProjectWindow", "Type"))
        self.open_dir_btn.setText(_translate("CreateProjectWindow", "..."))
        self.cancel_btn.setText(_translate("CreateProjectWindow", "Cancel"))
        self.create_project_btn.setText(_translate("CreateProjectWindow", "Create"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CreateProjectWindow = QtWidgets.QMainWindow()
    ui = Ui_CreateProjectWindow()
    ui.setupUi(CreateProjectWindow)
    CreateProjectWindow.show()
    sys.exit(app.exec_())
