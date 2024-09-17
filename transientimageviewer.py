from PyQt5.QtWidgets import  QSizePolicy, QCheckBox, QSpinBox, QPushButton, QWidget, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5 import QtCore

from transientgraphicsscene import ImageGraphicsScene
from transientgraphicsview import CustomGraphicsView


class ImageViewer(QWidget):
    
    sync_zoom_mouse = QtCore.pyqtSignal(float)
    sync_zoom_reset = QtCore.pyqtSignal()
    sync_scale_x = QtCore.pyqtSignal(float)
    sync_scale_y = QtCore.pyqtSignal(float)
    sync_change_mode = QtCore.pyqtSignal()
    
    def __init__(self,img,as_grayscale,lut,main_window):
        super().__init__()
        
        self.main_window = main_window
        self.cur_scale = [1 for i in range(2)]
        self.zoom_value_X = QSpinBox()
        self.zoom_value_X.setMinimum(10)
        self.zoom_value_X.setMaximum(500)
        self.zoom_value_X.setSingleStep(10)
        self.zoom_value_X.setValue(100)
        self.zoom_value_X.resize(0.1,self.zoom_value_X.height())
        self.zoom_value_X.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Preferred ))
        self.zoom_value_X.valueChanged.connect(self.ChangeScaleX)
        self.scale_slider_X = QSlider(Qt.Horizontal)
        self.scale_slider_X.setValue(100)
        self.scale_slider_X.setMinimum(10)
        self.scale_slider_X.setMaximum(500)
        self.scale_slider_X.valueChanged.connect(self.ChangeScaleX)
        self.button_reset = QPushButton("Reset Scale",self)
        self.button_reset.clicked.connect(self.ZoomReset)
        self.zoom_Y = QCheckBox("Zoom Y only")
        self.zoom_Y.setChecked(False)
        self.button_line = QPushButton("Space",self)
        self.button_line.clicked.connect(self.ChangeMode)
        #create layout and synchronize them
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.zoom_value_X)
        buttons_layout.addWidget(self.scale_slider_X)
        buttons_layout.addWidget(self.button_line)
        buttons_layout.addWidget(self.button_reset)
        buttons_layout.addWidget(self.zoom_Y)
        
        self.scene = ImageGraphicsScene(img,as_grayscale,lut,self.main_window)
        self.scene.mouse_zoom.connect(self.ZoomMouse)

        self.img = CustomGraphicsView(self.scene)
        self.img.setMouseTracking(True)

        
        self.zoom_value_Y = QSpinBox()
        self.zoom_value_Y.setMinimum(10)
        self.zoom_value_Y.setMaximum(500)
        self.zoom_value_Y.setSingleStep(10)
        self.zoom_value_Y.setValue(100)
        self.zoom_value_Y.resize(0.1,self.zoom_value_Y.height())
        self.zoom_value_Y.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Preferred ))
        self.zoom_value_Y.valueChanged.connect(self.ChangeScaleY)
        self.scale_slider_Y = QSlider(Qt.Vertical)
        self.scale_slider_Y.setMinimum(10)
        self.scale_slider_Y.setMaximum(500)
        self.scale_slider_Y.setValue(100)
        self.scale_slider_Y.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Preferred ))
        self.scale_slider_Y.valueChanged.connect(self.ChangeScaleY)
        layout_h = QHBoxLayout()
        layout_h.addWidget(self.img)
        layout_v = QVBoxLayout()
        layout_v.addWidget(self.zoom_value_Y)
        layout_v.addWidget(self.scale_slider_Y,Qt.AlignRight)
        layout_h.addLayout(layout_v)
        
        layout_submain = QVBoxLayout()
        layout_submain.addLayout(buttons_layout)
        layout_submain.addLayout(layout_h)
        layout = QVBoxLayout()
        layout.addLayout(layout_submain)
        self.setLayout(layout)
        
        
    def UpdateZooms(self):
        self.zoom_value_X.setValue(100*self.cur_scale[0])
        self.zoom_value_Y.setValue(100*self.cur_scale[1])
        self.scale_slider_X.setValue(100*self.cur_scale[0])
        self.scale_slider_Y.setValue(100*self.cur_scale[1])
        
    def ZoomMouseExt(self,increase_scale):
        if increase_scale:
            if not self.zoom_Y.isChecked():
                if self.cur_scale[0]+0.1 <= 5:
                    self.img.scale((self.cur_scale[0]+0.1)/self.cur_scale[0],1)
                    self.cur_scale[0] = self.cur_scale[0] + 0.1
                else:
                    self.img.scale(5/self.cur_scale[0],1)
                    self.cur_scale[0] = 5
            if self.cur_scale[1]+0.1 <= 5:
                self.img.scale(1,(self.cur_scale[1]+0.1)/self.cur_scale[1])
                self.cur_scale[1] = self.cur_scale[1] + 0.1
            else:
                self.img.scale(1,5/self.cur_scale[1])
                self.cur_scale[1] = 5
        else:
            if not self.zoom_Y.isChecked():
                if self.cur_scale[0]-0.1 >= 0.1:
                    self.img.scale((self.cur_scale[0]-0.1)/self.cur_scale[0],1)
                    self.cur_scale[0] = self.cur_scale[0] - 0.1
                else:
                    self.img.scale(0.1/self.cur_scale[0],1)
                    self.cur_scale[0] = 0.1
            if self.cur_scale[1] - 0.1 >= 0.1:
                self.img.scale(1,(self.cur_scale[1] - 0.1)/self.cur_scale[1])
                self.cur_scale[1] = self.cur_scale[1] - 0.1
            else:
                self.img.scale(1,0.1/self.cur_scale[1])
                self.cur_scale[1] = 0.1
        self.UpdateZooms()
    
    def ZoomMouse(self,increase_scale):
        self.ZoomMouseExt(increase_scale)
        self.sync_zoom_mouse.emit(increase_scale)
        self.img.CenterImage()
                
    def ZoomResetExt(self):
        self.img.scale(0.1/self.cur_scale[0],0.1/self.cur_scale[1])
        self.cur_scale = [0.1,0.1]
        self.UpdateZooms()
        
    def ZoomReset(self):
        self.ZoomResetExt()
        self.sync_zoom_reset.emit()
    
    def ChangeScaleXExt(self,value):
        self.img.scale(value/(100*self.cur_scale[0]),1)
        self.cur_scale[0] = value*1e-2
        self.UpdateZooms()
        
    def ChangeScaleX(self,value):
        self.ChangeScaleXExt(value)
        self.sync_scale_x.emit(value)
    
    def ChangeScaleYExt(self,value):
        self.img.scale(1,value/(100*self.cur_scale[1]))
        self.cur_scale[1] = value*1e-2
        self.UpdateZooms()
        
    def ChangeScaleY(self,value):
        self.ChangeScaleYExt(value)
        self.sync_scale_y.emit(value)
        
    def ChangeModeExt(self):
        text = self.button_line.text()
        if text == "Space":
            self.button_line.setText("Time")
            self.scene.mode = True
        else:
            self.button_line.setText("Space")
            self.scene.mode = False        
        
    def ChangeMode(self):
        self.ChangeModeExt()
        self.sync_change_mode.emit()