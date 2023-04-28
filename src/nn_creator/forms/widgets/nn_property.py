from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QLineEdit,
                             QComboBox, QSpinBox, QStyledItemDelegate, QCheckBox)
from nn_creator.forms.utils.event_filters import GlobalEventFilter2
from nn_creator.forms.utils.styles.colors import (DEFAULT_WHITE, NN_PROPERTY_BACKGROUND,
                                                  NN_PROPERTY_BACKGROUND_CHILD_ODD, NN_PROPERTY_BACKGROUND_CHILD_EVEN)
from nn_creator.forms.utils.resources import func_activations, immutable_fields

class LineEditWidget(QLineEdit):
    update_config_signal = pyqtSignal(dict)

    def __init__(self,
                 parent=None,
                 widget_id: int = None,
                 data=None,
                 prop_name: str = None,
                 items=None,
                 color: tuple = DEFAULT_WHITE):
        super().__init__(parent)
        self.widget_id = widget_id
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")
        self.setText(self.data)
        self.editingFinished.connect(self.update_data)

    def update_data(self):
        self.data = self.text()
        self.update_config_signal.emit({'prop_name': self.prop_name,
                                        'value': self.data,
                                        'widget_id': self.widget_id})

# class LineEditWidgetTuple(QLineEdit):
#     update_config_signal = pyqtSignal(dict)
#
#         def __init__(self,
#                  parent=None,
#                  widget_id: int = None,
#                  value=None,
#                  prop_name: str = None,
#                  color: tuple = DEFAULT_WHITE):
#         super().__init__(parent)
#         self.widget_id = widget_id
#         self.prop_name = prop_name
#         self.value = value
#         self.setStyleSheet(f"background-color: rgb{color};")
#         self.setText(self.data)
#         self.editingFinished.connect(self.update_data)
#
#             def update_data(self):
#         self.value = self.value()
#         self.update_config_signal.emit({'prop_name': self.prop_name,
#                                         'data': self.data,
#                                         'widget_id': self.widget_id})


class ComboBoxDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)


class ComboBoxWidget(QComboBox):
    update_config_signal = pyqtSignal(dict)

    def __init__(self,
                 parent=None,
                 widget_id: int = None,
                 data=None,
                 prop_name: str = None,
                 items: list = None,
                 color: tuple = DEFAULT_WHITE):
        super().__init__(parent)
        self.widget_id = widget_id
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")

        self.setItemDelegate(ComboBoxDelegate())
        self.addItems(items)
        # self.setCurrentIndex(0 if self.data else 1)
        self.currentTextChanged.connect(self.update_data)

    def update_data(self):
        self.data = self.currentText()
        self.update_config_signal.emit({'prop_name': self.prop_name,
                                        'value':  self.data,
                                        'widget_id': self.widget_id})


class SpinBoxWidgetInt(QSpinBox):
    update_config_signal = pyqtSignal(dict)

    def __init__(self,
                 parent=None,
                 widget_id: int = None,
                 data=None,
                 prop_name: str = None,
                 items=None,
                 color: tuple = DEFAULT_WHITE):
        super().__init__(parent)
        self.widget_id = widget_id
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")
        self.setMinimum(0)
        self.setMaximum(9999999)
        self.setValue(self.data)
        self.valueChanged.connect(self.update_data)

    def update_data(self):
        self.data = self.value()
        self.update_config_signal.emit({'prop_name': self.prop_name,
                                        'value': self.data,
                                        'widget_id': self.widget_id})


class CheckBoxWidget(QCheckBox):
    update_config_signal = pyqtSignal(dict)

    def __init__(self,
                 parent=None,
                 widget_id: int = None,
                 data=None,
                 prop_name: str = None,
                 items=None,
                 color: tuple = DEFAULT_WHITE):
        super().__init__(parent)
        self.widget_id = widget_id
        self.prop_name = prop_name
        self.data = data
        self.setStyleSheet(f"background-color: rgb{color};")
        self.setChecked(self.data)
        self.stateChanged.connect(self.update_data)

    def update_data(self):
        self.data = self.checkState()
        self.update_config_signal.emit({'prop_name': self.prop_name,
                                        'value': False if self.data == 0 else True,
                                        'widget_id': self.widget_id})

class NNPropertyWidget(QTreeWidget):
    # imm
    def __init__(self, parent=None, event_filter: GlobalEventFilter2 = None):
        super().__init__(parent=parent)
        self._event_filter = event_filter
        self.setHeaderLabels(['Property', "Value"])
        self.setStyleSheet("padding:0; margin:0;")
        self.moved_widget_id = None
        self.setStyleSheet(f"background-color: rgb{NN_PROPERTY_BACKGROUND};")
        self.property_items = {}
        self.update()

    @property
    def event_filter(self):
        return self._event_filter

    @event_filter.setter
    def event_filter(self, value: GlobalEventFilter2):
        self._event_filter = value

    def _display_properties(self, widget_id):
        widget = self.event_filter.nn_scheme_widgets[widget_id]
        self.clear()
        self.addTopLevelItem(QTreeWidgetItem(['Widget', widget.cfg['class_name']]))
        cou = 0
        for key, value in widget.cfg['config'].items():

            if key not in immutable_fields:
                if key == 'activation':
                    self._create_subwidget(widget_id=widget_id,
                                           cur_class=ComboBoxWidget,
                                           key=key,
                                           value=value,
                                           cou=cou,
                                           items=func_activations)
                elif isinstance(value, bool):
                    self._create_subwidget(widget_id, CheckBoxWidget, key, value, cou)
                elif isinstance(value, int):
                    self._create_subwidget(widget_id, SpinBoxWidgetInt, key, value, cou)
                elif isinstance(value, str):
                    self._create_subwidget(widget_id, LineEditWidget, key, value, cou)
                cou += 1
        #не работает
        # area_for_widget = self.topLevelItem(0)
        # area_for_widget.setExpanded(True)
        # temp = QTreeWidgetItem(area_for_widget)
        # area_for_widget.addChild(temp)
        # temp.setText(0, 'Immutable fields')
        # for key, value in widget.cfg['config'].items():
        #     if key in immutable_fields:
        #         color = NN_PROPERTY_BACKGROUND_CHILD_EVEN if cou % 2 == 0 else NN_PROPERTY_BACKGROUND_CHILD_ODD
        #         area_for_widget.addChild(temp)
        #         temp.setBackground(0, QColor(*color))
        #         temp.setBackground(1, QColor(*color))
        #         temp.setText(0, key)
        #         temp.setText(1, str(value))
        #         cou += 1

        self.update()

    def _create_subwidget(self, widget_id, cur_class, key, value, cou, items=None):
        color = NN_PROPERTY_BACKGROUND_CHILD_EVEN if cou % 2 == 0 else NN_PROPERTY_BACKGROUND_CHILD_ODD
        widget_type = cur_class(parent=self, widget_id=widget_id, data=value, prop_name=key, color=color, items=items)
        widget_type.update_config_signal.connect(self._display_updates)
        area_for_widget = self.topLevelItem(0)
        area_for_widget.setExpanded(True)
        temp = QTreeWidgetItem(area_for_widget)
        area_for_widget.addChild(temp)
        temp.setBackground(0, QColor(*color))
        temp.setText(0, key)
        self.setItemWidget(temp, 1, widget_type)

    def _display_updates(self, info):
        prop_name = info['prop_name']
        value = info['value']
        widget_id = info['widget_id']
        print('update:', widget_id, prop_name, value, type(value))
        widget = self.event_filter.nn_scheme_widgets[widget_id]
        widget.cfg['config'][prop_name] = value
        print(widget.cfg['config'][prop_name], type(widget.cfg['config'][prop_name]))

    def created_widget(self, widget):
        print('created_widget---')
        widget.mouse_press_signal.connect(self._display_properties)
        widget.delete_widget_signal.connect(self._clear_property_area)

    def _clear_property_area(self, widget_id):
        self.clear()

