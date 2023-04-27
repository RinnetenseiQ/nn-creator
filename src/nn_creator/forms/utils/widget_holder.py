import numpy as np

class WidgetHolder:
    def __init__(self, widgets=None):
        self.widgets = widgets if widgets else {}
        self.moved_widget_id = None
        self.last_deleted_widget = None
        self.last_created_widget = None

    def set_moved_widget_id(self, widget_id):
        self.moved_widget_id = widget_id

    def update_widgets_list(self, widget):
        key = np.max(list(self.widgets.keys())) + 1 if self.widgets else 1
        widget.widget_id = key
        widget.cast_id_signal.connect(self.set_moved_widget_id)
        widget.delete_widget_signal.connect(self.delete_widget_id)
        self.widgets[key] = widget
        self.set_moved_widget_id(key)
        self.last_created_widget = key
        # self.moved_widget_id = key
        print(f"event filter widgets_ids: {self.widgets.keys()}")

    def delete_widget_id(self, widget_id):
        print(f"delete: {widget_id}")
        self.widgets.pop(widget_id)
        self.last_deleted_widget = widget_id

