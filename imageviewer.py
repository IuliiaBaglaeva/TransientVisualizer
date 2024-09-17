from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QComboBox, QApplication, QSpinBox, QColorDialog, QTableWidget, QFileDialog, QListWidget, QLabel, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsLineItem, QTabWidget, QWidget, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QTransform
from PyQt5.QtCore import Qt

class ImageViewer(QWidget):
    
    def __init__(self,img):
        super().__init__()
        
        self.cur_scale = 1
        self.button1 = QPushButton("Zoom +",self)
        self.button1.clicked.connect(self.ZoomPlus)
        self.button2 = QPushButton("Zoom -",self)
        self.button2.clicked.connect(self.ZoomMinus)
        self.button3 = QPushButton("Reset Scale",self)
        self.button3.clicked.connect(self.ZoomReset)
        self.button_line = QPushButton("Vertical",self)
        self.button_line.clicked.connect(self.ChangeMode)
        self.scale_label = QLabel(f"{self.cur_scale*100}%")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setValue(100)
        self.scale_slider.setMinimum(10)
        self.scale_slider.setMaximum(400)
        self.scale_slider.valueChanged.connect(self.ChangeScale)
        self.line_label = QLabel("Line Width",self)
        self.width_value = QSpinBox()
        self.width_value.setValue(2)
        self.width_value.setMinimum(1)
        #create layout and synchronize them
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.scale_label)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.scale_slider)
        buttons_layout.addWidget(self.button1)
        buttons_layout.addWidget(self.button3)
        buttons_layout.addWidget(self.button_line)
        buttons_layout.addWidget(self.line_label)
        buttons_layout.addWidget(self.width_value)
        
        self.scene = ImageGraphicsScene()
        im_new = np.floor((img - np.min(img))/(np.max(img) - np.min(img))*255 + 0.5)
        im_new = im_new.astype("uint8")

        self.scene.addImage(im_new)
        self.img = QGraphicsView(self.scene)
        
        layout_but_graph = QVBoxLayout()
        layout_but_graph.addLayout(buttons_layout)
        layout_but_graph.addWidget(self.img)
        layout = QVBoxLayout()
        layout.addLayout(layout_but_graph)
        self.setLayout(layout)
        
    def UpdateText(self):
        self.scale_label.setText(f"{int(self.cur_scale*100)}%")
        
    def ZoomPlus(self):
        self.img.scale(1 + 0.1/self.cur_scale,1 + 0.1/self.cur_scale)
        self.cur_scale += 0.1
        self.scale_slider.setValue(self.cur_scale*100)
        self.UpdateText()
        
    def ZoomMinus(self):
        if self.cur_scale > 0.11:
            self.img.scale(1 - 0.1/self.cur_scale,1 - 0.1/self.cur_scale)
            self.cur_scale -= 0.1
            self.scale_slider.setValue(self.cur_scale*100)
            self.UpdateText()
        
    
    def ZoomReset(self):
        self.img.scale(1/self.cur_scale,1/self.cur_scale)
        self.cur_scale = 1
        self.scale_slider.setValue(self.cur_scale*100)
        self.UpdateText()
    
    def ChangeScale(self,value):
        self.img.scale(value/(100*self.cur_scale),value/(100*self.cur_scale))
        self.cur_scale = value*1e-2
        self.UpdateText()
    
    def ChangeMode(self):
        text = self.button_line.text()
        if text == "Vertical":
            self.button_line.setText("Horizontal")
            self.scene.mode = True
        else:
            self.button_line.setText("Vertical")
            self.scene.mode = False