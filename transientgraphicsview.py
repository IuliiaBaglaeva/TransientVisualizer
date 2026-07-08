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
        if self.scene() is not None and hasattr(self.scene(), "qImg"):
            self.sc_pos = self.mapToScene(QPoint(event.x(),event.y()))
            img_coord = self.scene().qImg.mapFromScene(self.sc_pos)
            x = min(max(img_coord.x(), 0), self.scene().img.shape[1])
            y = min(max(img_coord.y(), 0), self.scene().img.shape[0])
            self.mouse_coord.emit(int(x),int(y))

    def CenterImage(self):
        self.centerOn(self.sc_pos.toPoint())
        QCursor.setPos(self.mapToGlobal(self.geometry().center()))
                
