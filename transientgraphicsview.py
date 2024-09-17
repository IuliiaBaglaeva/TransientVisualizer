from PyQt5 import QtCore
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import  QGraphicsView
from PyQt5.QtGui import QCursor

class CustomGraphicsView(QGraphicsView):
    
    mouse_coord = QtCore.pyqtSignal(int,int)
    mouse_zoom = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.sc_pos = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        item = self.itemAt(QPoint(event.x(),event.y()))
        if item is not None:
            self.sc_pos = self.mapToScene(QPoint(event.x(),event.y()))
            img_coord = item.mapFromScene(self.sc_pos.toPoint())
            self.mouse_coord.emit(int(img_coord.x()),int(img_coord.y()))

    def CenterImage(self):
        self.centerOn(self.sc_pos.toPoint())
        QCursor.setPos(self.mapToGlobal(self.geometry().center()))
                