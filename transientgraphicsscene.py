from PyQt5 import QtCore
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsLineItem
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QTransform
from PyQt5.QtCore import Qt
import numpy as np

class ImageGraphicsScene(QGraphicsScene):
    
    new_horizontal_line_created = QtCore.pyqtSignal(QGraphicsLineItem)
    new_vertical_lines_created = QtCore.pyqtSignal(QGraphicsLineItem,QGraphicsLineItem)
    line_is_moved = QtCore.pyqtSignal(QGraphicsLineItem,int,int)
    image_position_changed = QtCore.pyqtSignal(int,int)
    mouse_zoom = QtCore.pyqtSignal(bool)
    
    def __init__(self, img,as_grayscale,lut,main_window,parent=None): 
        QGraphicsScene.__init__(self, parent)
        self.main_window = main_window
        self.ctrl_pressed = False
        self.img = np.copy(img)
        self.is_gray = as_grayscale
        self.qImg = ""
        self.lut = np.copy(lut)
        self.height, self.width = img.shape
        self.bytesPerLine = self.width
        if as_grayscale:
            self.qImg = QGraphicsPixmapItem(QPixmap(QImage(self.img.data, self.width, self.height, self.bytesPerLine, QImage.Format_Grayscale8)))
        else:
            qI = QImage(self.img.data, self.width, self.height, self.bytesPerLine, QImage.Format_Indexed8)
            qI.setColorTable(lut)
            self.qImg = QGraphicsPixmapItem(QPixmap(qI))
        self.qImg.setTransformationMode(Qt.SmoothTransformation)
        self.qImg.setFlag(QGraphicsItem.ItemIsMovable)
        self.addItem(self.qImg)
        self.mode = False #0 - vertical, 1 - horizontal
        self.lines = []
        
        self.v_pen = None #pen for vertical line
        self.moving_line = None #temporary line for the movement of line
        self.mouse_pressed = False

    def wheelEvent(self,event):
        if self.ctrl_pressed:
            event.accept()
            if event.delta() > 0:
                self.mouse_zoom.emit(True)
            else:
                self.mouse_zoom.emit(False)    
                
    def setOption(self, opt):
        self.opt = opt

    def addImage(self,img):
        self.img = np.copy(img)
        height, width = img.shape
        bytesPerLine = width
        qImg = QGraphicsPixmapItem(QPixmap(QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)))
        self.addItem(qImg)
          
    def mousePressEvent(self, event):
        self.mouse_pressed = True
        if not self.ctrl_pressed:
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, QGraphicsLineItem):
               self.moving_line = item
            else:
                super().mousePressEvent(event)
        else:
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, QGraphicsLineItem):
                return
            color = np.random.randint(0,256,3)
            pen = QPen(QColor(*color,255),15, QtCore.Qt.SolidLine)
            x = event.scenePos().x() - self.qImg.scenePos().x()
            y = event.scenePos().y() - self.qImg.scenePos().y()
            if self.mode:
                if y < 0:
                    self.lines.append(QGraphicsLineItem(0, 0, self.img.shape[1],0,self.qImg))  
                elif y > self.img.shape[0]:
                    self.lines.append(QGraphicsLineItem(0, self.img.shape[0], self.img.shape[1],self.img.shape[0],self.qImg))
                else:
                    self.lines.append(QGraphicsLineItem(0, y, self.img.shape[1], y, self.qImg))
                self.lines[-1].setPen(pen)
                self.new_horizontal_line_created.emit(self.lines[-1])
            else:
                if x < 0:
                    self.lines.append(QGraphicsLineItem(0, 0, 0, self.img.shape[0],self.qImg))
                elif x > self.img.shape[1]:
                    self.lines.append(QGraphicsLineItem(self.img.shape[1], 0, self.img.shape[1], self.img.shape[0], self.qImg))
                else:
                    self.lines.append(QGraphicsLineItem(x, 0, x, self.img.shape[0], self.qImg))
                self.lines[-1].setPen(pen)
                self.v_pen = pen
    
    
    def MoveLine(self,line,x,y):
        if line.line().x1() == line.line().x2():
            if x < 0:
                line.setLine(0, 0, 0, self.img.shape[0])
            elif x > self.img.shape[1]:
                line.setLine(self.img.shape[1], 0, self.img.shape[1], self.img.shape[0])
            else:
                line.setLine(x, 0, x, self.img.shape[0])
        else:
            if y < 0:
                line.setLine(0, 0, self.img.shape[1],0)  
            elif y > self.img.shape[0]:
                line.setLine(0, self.img.shape[0], self.img.shape[1],self.img.shape[0])
            else:
                line.setLine(0, y, self.img.shape[1],y)      
        
    def mouseMoveEvent(self, event):
        if self.moving_line:
            x = event.scenePos().x() - self.qImg.scenePos().x()
            y = event.scenePos().y() - self.qImg.scenePos().y()
            self.MoveLine(self.moving_line,x,y)
            self.line_is_moved.emit(self.moving_line,x,y)
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        self.mouse_pressed = False
        self.moving_line = None
        if self.v_pen:
            x = event.scenePos().x() - self.qImg.scenePos().x()
            self.lines.append(QGraphicsLineItem(x, 0, x, self.img.shape[0], self.qImg))
            self.lines[-1].setPen(self.v_pen)
            self.new_vertical_lines_created.emit(self.lines[-2],self.lines[-1])
            self.v_pen = None
        else:
            super().mouseReleaseEvent(event)
            self.image_position_changed.emit(self.qImg.scenePos().x(),self.qImg.scenePos().y())
            
    def changeImgPosition(self,x,y):
        self.qImg.setPos(x,y)
            
    def externalAddHLine(self,line): #adding the horizontal line based on the signal from different scene
        self.lines.append(QGraphicsLineItem(line.line(), self.qImg))
        self.lines[-1].setPen(line.pen())
        return self.lines[-1]
        
    def externalAddVLines(self,line,line2): #adding the vertical lines based on the signal from different scene
        return (self.externalAddHLine(line),self.externalAddHLine(line2))
    
    def externalMoveLine(self,pars): #moving the line based on the signal from different scene
        item = self.itemAt(*pars[:2], QTransform())
        if isinstance(item, QGraphicsLineItem):
            if pars[4]:
                item.setLine(0,pars[3],self.img.shape[1],pars[3])
            else:
                item.setLine(pars[2], 0, pars[2], self.img.shape[0])

    def CreateHLine(self,y):
        color = np.random.randint(0, 256, 3)
        pen = QPen(QColor(*color, 255), 15, QtCore.Qt.SolidLine)
        y = y - self.qImg.scenePos().y()
        self.lines.append(QGraphicsLineItem(0, y, self.img.shape[1], y, self.qImg))
        self.lines[-1].setPen(pen)
        return self.lines[-1]

    def CreateVLines(self,x1,x2):
        color = np.random.randint(0, 256, 3)
        pen = QPen(QColor(*color, 255), 15, QtCore.Qt.SolidLine)
        x1 = x1 - self.qImg.scenePos().x()
        self.lines.append(QGraphicsLineItem(x1, 0, x1, self.img.shape[0], self.qImg))
        self.lines[-1].setPen(pen)
        x2 = x2 - self.qImg.scenePos().x()
        self.lines.append(QGraphicsLineItem(x2, 0, x2, self.img.shape[0], self.qImg))
        self.lines[-1].setPen(pen)
        return [self.lines[-2],self.lines[-1]]
