from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QLineEdit
from nn_creator.forms.utils.event_filters import GlobalEventFilter2


class EditWidget(QLineEdit):
    update_config_signal = pyqtSignal(dict)

    def __init__(self, data, parent=None, prop_name: str = None):
        super().__init__(parent)
        self.prop_name = prop_name
        self.data = data
        # Создаем поле ввода для редактирования данных
        self.setText(data)
        self.editingFinished.connect(self.update_data)

    def update_data(self):
        # Обновляем данные при изменении поля ввода
        self.data = self.text()
        # print(f'update_data:{self.data}')
        self.update_config_signal.emit({self.prop_name: self.data})


class NNPropertyWidget(QTreeWidget):
    def __init__(self, parent=None, event_filter: GlobalEventFilter2 = None):
        super().__init__(parent=parent)
        self._event_filter = event_filter
        # self.setObjectName("property_area")
        # self.setFixedSize(300, 300)
        # self.setAcceptDrops(True)
        self.setHeaderLabels(['Property', "Value"])
        self.setStyleSheet("padding:0; margin:0;")
        self.moved_widget_id = None
        self.setStyleSheet("background-color:green;")
        # self.update()
        self.property_items = {}
        # self.setDragEnabled(True)
        self.update()

    @property
    def event_filter(self):
        return self._event_filter

    @event_filter.setter
    def event_filter(self, value: GlobalEventFilter2):
        self._event_filter = value

    # TODO: add checker for data type
    def display_properties(self, widget_id):
        widget = self.event_filter.nn_scheme_widgets[widget_id]
        self.clear()
        # combo = QComboBox()
        # combo.addItems(["Option 1", "Option 2", "Option 3"])
        # combo.setCurrentIndex(0)
        #
        # self.combo = QLineEdit('dfghjk')
        # self.combo.editingFinished.connect(self.update_data)

        self.addTopLevelItem(QTreeWidgetItem(['Widget', widget.cfg['class_name']]))

        # area_about_widget = self.topLevelItem(0)
        # temp = QTreeWidgetItem(area_about_widget)
        # area_about_widget.addChild(temp)
        #
        # temp.setText(0, 'gggg')
        # self.setItemWidget(temp, 1, self.combo)
        # temp.setBackground(1, QColor(255, 0, 0))

        for key, value in widget.cfg['config'].items():
            line_edit = EditWidget(parent=self, data=str(value), prop_name=key)
            line_edit.update_config_signal.connect(self.display_updates)
            area_about_widget = self.topLevelItem(0)
            temp = QTreeWidgetItem(area_about_widget)
            area_about_widget.addChild(temp)
            temp.setText(0, key)
            self.setItemWidget(temp, 1, line_edit)

            # area_about_widget = self.topLevelItem(0)
            # area_about_widget.addChild(QTreeWidgetItem(['ddd']))

        self.update()

    def display_updates(self, data):
        key = list(data.keys())
        value = list(data.values())
        print('update:', key[0], value[0])

    def created_widget(self, widget):
        print('created_widget---')
        widget.mouse_press_signal.connect(self.display_properties)
