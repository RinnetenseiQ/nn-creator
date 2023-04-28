from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QLineEdit,
                             QComboBox, QSpinBox, QStyledItemDelegate, QStyle)
from nn_creator.forms.utils.event_filters import GlobalEventFilter2


class LineEditWidget(QLineEdit):
    update_config_signal = pyqtSignal(dict)

    def __init__(self, data, parent=None, prop_name: str = None, color: tuple = (255, 255, 255)):
        super().__init__(parent)
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")
        self.setText(self.data)
        self.editingFinished.connect(self.update_data)

    def update_data(self):
        self.data = self.text()
        self.update_config_signal.emit({self.prop_name: str(self.data)})


# class LineEditWidgetTuple(QLineEdit):
#     update_config_signal = pyqtSignal(dict)
#
#     def __init__(self, data, parent=None, prop_name: str = None, color: tuple = (255, 255, 255)):
#         super().__init__(parent)
#         self.prop_name = prop_name
#         self.data = data
#         self.setStyleSheet(f"background-color: rgb{color};")
#         self.setText(self.data)
#         self.editingFinished.connect(self.update_data)
#
#     def update_data(self):
#         self.data = self.text()
#         self.update_config_signal.emit({self.prop_name: str(self.data)})

# работает как нужно
class ComboBoxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # if option.state & QStyle.State_MouseOver:
        #     # Определяем цвет фона для элемента при наведении на него мышью
        #     background_color = QColor(255,255,255)
        # else:
        #     background_color = QColor(255,255,255)
        #
        # # Рисуем фон элемента
        # palette = QPalette()
        # palette.setColor(QPalette.Base, background_color)
        # option.palette = palette
        super().paint(painter, option, index)
class ComboBoxWidget(QComboBox):
    update_config_signal = pyqtSignal(dict)

    def __init__(self, data, parent=None, prop_name: str = None, color: tuple = (255, 255, 255)):
        super().__init__(parent)
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")

        self.setItemDelegate(ComboBoxDelegate())
        self.addItems(['True', 'False'])
        self.setCurrentIndex(0 if self.data else 1)
        self.currentTextChanged.connect(self.update_data)

    def update_data(self):
        self.data = self.currentText()
        self.update_config_signal.emit({self.prop_name: True if self.data == 'True' else False})


class SpinBoxWidgetInt(QSpinBox):
    update_config_signal = pyqtSignal(dict)

    def __init__(self, data, parent=None, prop_name: str = None, color: tuple = (255, 255, 255)):
        super().__init__(parent)
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")
        self.setMinimum(0)
        self.setMaximum(9999999)
        self.setValue(self.data)
        self.valueChanged.connect(self.update_data)

    def update_data(self):
        self.data = self.value()
        self.update_config_signal.emit({self.prop_name: self.data})


class NNPropertyWidget(QTreeWidget):
    def __init__(self, parent=None, event_filter: GlobalEventFilter2 = None):
        super().__init__(parent=parent)
        self._event_filter = event_filter
        self.setHeaderLabels(['Property', "Value"])
        self.setStyleSheet("padding:0; margin:0;")
        self.moved_widget_id = None
        self.setStyleSheet("background-color: rgb(255, 255, 191);")
        self.property_items = {}
        self.update()

    @property
    def event_filter(self):
        return self._event_filter

    @event_filter.setter
    def event_filter(self, value: GlobalEventFilter2):
        self._event_filter = value

    # TODO: add tuple checker for data type
    # TODO: maybe "acivation" move to combo box?
    def display_properties(self, widget_id):
        widget = self.event_filter.nn_scheme_widgets[widget_id]
        self.clear()
        self.addTopLevelItem(QTreeWidgetItem(['Widget', widget.cfg['class_name']]))
        cou = 0
        for key, value in widget.cfg['config'].items():
            cou += 1
            # print('type:', type(value))
            #TODO: can make a bool with a checkbox?
            if isinstance(value, bool):
                self._create_widget(ComboBoxWidget, key, value, cou)
            elif isinstance(value, int):
                self._create_widget(SpinBoxWidgetInt, key, value, cou)
            # elif isinstance(value, (tuple, list, set)):
            #     self._create_widget(LineEditWidgetTuple, key, value, cou)
            elif isinstance(value, str):
                self._create_widget(LineEditWidget, key, value, cou)

        self.update()

    def _create_widget(self, cur_class, key, value, cou):
        color = (255, 255, 191) if cou % 2 == 0 else (255, 255, 222)
        widget_type = cur_class(parent=self, data=value, prop_name=key, color=color)
        widget_type.update_config_signal.connect(self.display_updates)
        area_about_widget = self.topLevelItem(0)
        area_about_widget.setExpanded(True)
        temp = QTreeWidgetItem(area_about_widget)
        area_about_widget.addChild(temp)
        temp.setBackground(0, QColor(*color))
        temp.setText(0, key)
        self.setItemWidget(temp, 1, widget_type)

    def display_updates(self, data):
        key = list(data.keys())
        value = list(data.values())
        print('update:', key[0], value[0], type(value[0]))

    def created_widget(self, widget):
        print('created_widget---')
        widget.mouse_press_signal.connect(self.display_properties)
