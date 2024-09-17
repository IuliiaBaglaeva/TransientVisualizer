import pandas as pd
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtWidgets import qApp, QMainWindow, QErrorMessage, QDoubleSpinBox, QSpinBox, QTableWidgetItem,QMessageBox, QAction, QComboBox, QApplication, QColorDialog, QTableWidget, QFileDialog, QLabel, QPushButton, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtGui import qRgb, QColor, QPen
import pyqtgraph as pg
import sys
import os
import tifffile
import xml.etree.ElementTree as ET
from enum import IntFlag
from copy import deepcopy
from scipy import signal, fft
import json
import multiprocessing
from functools import partial


# custom modules for UI inherited from PyQt Widgets
from transientimageviewer import ImageViewer

# calculations libs
import numpy as np


class LineType(IntFlag):
    Background = 0
    ActiveSignal = 1
    EdgeorArtifact = 2
    TimeSegment = 3

class PlacedWidget(IntFlag):
    Fluo = 0
    Sarcomere = 1
    Both = 2
    
class Line():
    def __init__(self):
        self.Lines = [None, None]
        self.LineType = None
        self.Widget = None

# mode: 0 - vertical, 1 - horizontal

def process_element(args):
    i, im, roi, n_lines, dx, lambda_max, lambda_min, l_max, l_min = args
    T = dx * 1e-3
    slen = np.zeros(n_lines)
    for i in range(n_lines):
        im_i = im[roi == i + 1]
        N = im_i.shape[0]
        window = signal.windows.hann(N)
        y_f = fft.fft(im_i * window / np.sum(window))
        x_f = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        X_f = x_f[(x_f >= lambda_max) & (x_f <= lambda_min)]
        l_f = np.abs(y_f[:N // 2])[(x_f >= lambda_max) & (x_f <= lambda_min)]
        idx_peak = np.argmax(l_f)
        if  0 < idx_peak < len(l_f) - 1:
            lp1 = X_f[idx_peak + 1]
        elif idx_peak == 0:
            slen[i] = l_max
            continue
        else:
            slen[i] = l_min
            continue
        lm1 = X_f[idx_peak - 1]
        l0 = X_f[idx_peak]
        A = np.array([[lm1 * lm1, lm1, 1],
                      [l0 * l0, l0, 1],
                      [lp1 * lp1, lp1, 1]])
        Y = np.array([l_f[idx_peak - 1], l_f[idx_peak], l_f[idx_peak + 1]])
        coefs = np.linalg.solve(A, Y)
        s = 1 / (-coefs[1] / (2 * coefs[0]))
        if s > l_max:
            slen[i] = l_max
        elif s < l_min:
            slen[i] = l_min
        else:
            slen[i] = s
    return np.mean(slen)


# Class of the window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Load the UI Page and init class fields
        uic.loadUi('window.ui', self)
        self.path = ""
        self.file_name = ""
        self.image_name = ""
        self.fluo_image = None
        self.trans_image = None
        self.h_lines = []
        self.v_lines = []
        self.time_points = ["Start","End"]
        self.dx = 0
        self.dt = 0
        # GUI Widgets
        self.status = self.findChild(QLabel, 'StatusLabel')
        self.coord_label = self.findChild(QLabel, 'CoordLabel') 
        self.img_viewer = self.findChild(QTabWidget,'ChannelsTabs')
        self.dt_spinbox = self.findChild(QDoubleSpinBox,'TimeStep')
        self.dt_spinbox.valueChanged.connect(self.Changedt)
        self.dx_spinbox = self.findChild(QDoubleSpinBox,'SpaceStep')
        self.dx_spinbox.valueChanged.connect(self.Changedx)
        # Table
        self.line_table = self.findChild(QTableWidget,'Linescans')
        line_tab = self.findChild(QWidget,'tab1')
        self.line_table.cellClicked.connect(self.RunColorDialogV)
        layout = QVBoxLayout()
        layout.addWidget(self.line_table)
        line_tab.setLayout(layout)
        self.time_table = self.findChild(QTableWidget,'TimePoints')
        self.time_table.cellClicked.connect(self.RunColorDialogH)
        self.time_table.cellChanged.connect(self.ChangeTimeComboBox)
        time_tab = self.findChild(QWidget,'tab2')
        layout = QVBoxLayout()
        layout.addWidget(self.time_table)
        time_tab.setLayout(layout)
        #GUI Buttons
        self.open_qbutton = self.findChild(QAction, 'actionOpen_Image')
        self.open_qbutton.triggered.connect(self.OpenImage)
        self.actionSave_Project.triggered.connect(self.SaveProject)
        self.actionOpen_Project.triggered.connect(self.OpenProject)
        self.change_y_only = False
        # Computation buttons
        self.ComputeFluo.clicked.connect(self.ComputeFluoProfile)
        self.ComputeLineScan.clicked.connect(self.ShowSarcoLineScan)
        self.ComputeLength.clicked.connect(self.ComputeSarProfile)
        self.ComputeArtifacts.clicked.connect(self.ComputeFluoArtifacts)
        # Outliers buttons
        self.OutlierStartButton.clicked.connect(self.ComputeSarcowoOutliers)
        self.PickOutlierColorButton.clicked.connect(self.PickColorForOutlierPlot)
        self.OutlierDescription.clicked.connect(self.ShowOutlierDescriptionWindow)
        # output parameters
        self.t_F = None
        self.t_L = None
        self.t_L_no_local_outliers = None
        self.outlier_plot = None
        self.F = None
        self.L = None
        self.L_no_local_outliers = None
        self.n_outliers = 0
        self.outlier_pen = None
        self.bg_mean = None
        self.bg_std = None
        #set random color for outlier
        self.SetColorForOutlierPen(QColor(*np.random.randint(0,256,3),255))
        #set mouse methods for plots
        self.setMouseTracking(True)
        self.LinePlot.scene().sigMouseMoved.connect(partial(self.MouseMovedonPlot, widget=self.LinePlot, label = self.FluoCoordinates))
        self.LineScanPlot.scene().sigMouseMoved.connect(partial(self.MouseMovedonPlot, widget=self.LineScanPlot, label = self.IntensityCoordinates))
        self.SarPlot.scene().sigMouseMoved.connect(partial(self.MouseMovedonPlot, widget=self.SarPlot, label = self.SarCoordinates))
        self.ArtifactsPlot.scene().sigMouseMoved.connect(partial(self.MouseMovedonPlot, widget=self.ArtifactsPlot, label = self.ArtifactsCoordinates))

    def SetColorForOutlierPen(self, color):
        self.OutlierColorLabel.setStyleSheet(f"background-color: {color.name()};")
        self.outlier_pen = pg.mkPen(color=color)

    def PickColorForOutlierPlot(self):
        color_picked = QColorDialog.getColor()
        self.SetColorForOutlierPen(color_picked)

    def ShowOutlierDescriptionWindow(self):
        # Create a message box
        msg = QMessageBox()
        msg.setWindowTitle("Information")

        # Create a QLabel to display the text
        label = QLabel("The software is capable of eliminating distinct outliers from the time series - "
                       "these are isolated points that deviate from the standard sarcomere length curve due to FFT noise. "
                       "Rather than filtering these points, the software removes them entirely. This approach is designed "
                       "to ensure that the kinetics of the process are minimally affected. The outlier removal process "
                       "involves two stages:\n\n"
                       "1. Removal of points that fall outside of a user-defined range.\n"
                       "2. Calculation of the Z-score for each point within a centered window of a user-defined size. "
                       "A point is defined as an outlier if its Z-score exceeds 3.")
        label.setWordWrap(True)
        label.setMinimumSize(QtCore.QSize(480, 0))  # Set the minimum width of the QLabel
        # Add the QLabel to the message box
        msg.layout().addWidget(label)
        msg.exec()

    def centered_rolling_mean(self,data, window_size):
        # Extend the data at both ends with mirrored copies to handle the boundaries
        extended_data = np.concatenate([data[window_size-2::-1], data, data[:-window_size:-1]])
    
        # Apply the convolution with a uniform window
        smoothed = np.convolve(extended_data, np.ones(window_size), mode='valid') / window_size
    
        # Discard the extra data at the ends and return the smoothed data
        return smoothed[window_size // 2:-(window_size // 2)]

    def ComputeSarcowoOutliers(self):
        if self.L is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("You need to calculate the sarcomere first!")
            error_dialog.exec_()
            return
        y_lower_bound = self.MinimumOutlierSpinBox.value()
        y_upper_bound = self.MaximumOutlierSpinBox.value()
        self.L_no_local_outliers = self.L[(self.L >= y_lower_bound) & (self.L <= y_upper_bound)]
        self.t_L_no_local_outliers = self.t_L[(self.L >= y_lower_bound) & (self.L <= y_upper_bound)]
        window_size = self.WindowSizeOutlierSpinBox.value()
        # Calculate the centered rolling mean
        rolling_mean  = self.centered_rolling_mean(self.L_no_local_outliers, window_size)

        # Calculate the absolute difference between the actual value and the rolling mean
        diff = np.abs(self.L_no_local_outliers - rolling_mean)

        # Define a threshold for which differences will be considered as outliers
        threshold = 3 * diff.std()

        # Identify the local outliers
        outliers = diff > threshold

        # Remove the local outliers
        self.L_no_local_outliers = self.L_no_local_outliers[~outliers]
        self.t_L_no_local_outliers = self.t_L_no_local_outliers[~outliers]
        self.n_outliers = self.t_L.shape[0] - self.t_L_no_local_outliers.shape[0]
        print(self.outlier_plot)
        if self.outlier_plot is not None:
            self.SarPlot.removeItem(self.outlier_plot)
        self.outlier_plot = self.SarPlot.plot(self.t_L_no_local_outliers,self.L_no_local_outliers,pen=self.outlier_pen)

    def MouseMovedonPlot(self, evt, widget, label):
        if widget.plotItem.vb.mapSceneToView(evt):
            point = widget.plotItem.vb.mapSceneToView(evt)
            label.setText(f"x: {point.x():.3f}, y: {point.y():.3f}")

    def ClearAll(self):
        self.v_lines.clear()
        self.h_lines.clear()
        self.time_table.setRowCount(0)
        self.line_table.setRowCount(0)
        self.StartComboBox.clear()
        self.StartComboBox.addItems(["Start"])
        self.EndComboBox.clear()
        self.EndComboBox.addItems(["End"])
        self.FFTComboBox.clear()
        self.SarPlot.clear()
        self.LineScanPlot.clear()
        self.LinePlot.clear()
        self.ArtifactsPlot.clear()
        self.t_F = None
        self.t_L = None
        self.L_no_local_outliers = None
        self.t_L_no_local_outliers = None
        self.n_outliers = 0
        self.F = None
        self.L = None

    def closeEvent(self,event):#
        exit_msg = "Are you sure you want to exit?"
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Exit warning")
        msg_box.setText(f"{exit_msg}")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        resp = msg_box.exec()
        if resp == QMessageBox.Yes:
            self.close()
        else:
            event.ignore()


    def OpenProject(self):
        filename = QFileDialog.getOpenFileName(self, "Open the project", "", "Calcium and Sarcomere extraction files (*.casar)")[0]
        if filename != "":
            self.path, name = os.path.split(os.path.abspath(filename))
            f = open(filename)
            project_data = json.load(f)
            f.close()
            # add image
            self.ClearAll()
            self.file_name = project_data["filename"]
            tif = tifffile.TiffFile(f"{self.path}\{self.file_name}")
            self.image_name, ext = os.path.splitext(self.file_name)
            while ext:
                self.image_name, ext = os.path.splitext(self.image_name)
            self.fluo_image = tif.pages[0].asarray()
            self.trans_image = tif.pages[1].asarray()
            metadata = tif.ome_metadata
            tree = ET.fromstring(metadata)
            self.dx = float(tree.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')[0].attrib["PhysicalSizeX"]) # 'cause usually in micrometers
            self.dt = float(tree.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')[0].attrib["PhysicalSizeY"]) # 'cause usually in seconds
            self.dt_spinbox.setValue(self.dt)
            self.dx_spinbox.setValue(self.dx)
            self.status.setText("Current Image: " + f"{self.path}\{self.file_name}")
            self.time_points = ["Start","End"]
            self.SetImageViewer()
            # add lines
            for l in project_data["Time_lines"]:
                lines = Line()
                lines.Lines[0] = self.img_viewer.widget(0).scene.CreateHLine(l["t"])
                lines.Lines[1] = self.img_viewer.widget(1).scene.externalAddHLine(lines.Lines[0])
                lines.LineType = LineType.TimeSegment
                lines.Widget = PlacedWidget.Both
                self.h_lines.append(lines)
                self.addTimesRow()
                rowPosition = self.time_table.rowCount()
                self.time_table.item(rowPosition - 1, 0).setText(l["Name"])
                self.StartComboBox.addItem(l["Name"])
                self.FFTComboBox.addItem(l["Name"])
                self.EndComboBox.addItem(l["Name"])
            for l in project_data["Space_lines"]:
                lines = Line()
                lines.Widget = PlacedWidget(l["Place"])
                lines.LineType = LineType(l["Role"])
                lines.Lines = self.img_viewer.widget(lines.Widget).scene.CreateVLines(l["x1"],l["x2"])
                self.v_lines.append(lines)
                self.addLineScansRow()
                rowPosition = self.line_table.rowCount()
                self.line_table.item(rowPosition - 1, 0).setText(l["Name"])
            # check for outliers. If does not exist, set up default values
            if "Outliers" in project_data:
                self.MinimumOutlierSpinBox.setValue(project_data["Outliers"]["Min length"])
                self.MaximumOutlierSpinBox.setValue(project_data["Outliers"]["Max length"])
                self.WindowSizeOutlierSpinBox.setValue(project_data["Outliers"]["Window size"])

    def SaveProject(self):
        if self.t_F is None or self.t_L is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("Both  sarcomere and fluorescence must be computed to save the project")
            error_dialog.exec_()
            return
        if self.t_F.shape != self.t_L.shape or not np.all(np.abs(self.t_F - self.t_L) < 1e-7):
            error_dialog = QErrorMessage()
            error_dialog.showMessage("The sarcomere and fluorescence must have the same time range")
            error_dialog.exec_()
            return
        idx_project = None
        idx_project_int = 0
        while(os.path.exists(f"{self.path}\{self.image_name}{idx_project or ''}.casar")):
            if idx_project is None:
                idx_project_int = 1
            else:
                idx_project_int += 1
            idx_project = f"_{idx_project_int}"
        cols = ["time, ms","F, AU", "L, um"]
        data_F = np.zeros((self.t_F.shape[0],2))
        data_F[:,0] = self.t_F
        data_F[:,1] = self.F
        dict_bg = dict(Background_mean=self.bg_mean, Background_std=self.bg_std)
        data_bg = pd.DataFrame(dict_bg, index=[0])
        writer = pd.ExcelWriter(f"{self.path}\{self.image_name}{idx_project or ''}_F.xlsx", engine='xlsxwriter')
        # Write each dataframe to a different file.
        cols = ["time, ms","F, AU"]
        data = pd.DataFrame(data_F,columns=cols)
        data.to_excel(writer, sheet_name='Calcium', index = False)
        data_bg.to_excel(writer, sheet_name='BG Info', index = False)
        writer.save()
        if self.outlier_plot is None:
            data_L = np.zeros((self.t_L.shape[0],2))
            data_L[:,0] = self.t_L
            data_L[:,1] = self.L
        else:
            data_L = np.zeros((self.t_L_no_local_outliers.shape[0],2))
            data_L[:,0] = self.t_L_no_local_outliers
            data_L[:,1] = self.L_no_local_outliers
        cols = ["time, ms", "L, um"]
        data = pd.DataFrame(data_L,columns=cols)
        writer = pd.ExcelWriter(f"{self.path}\{self.image_name}{idx_project or ''}_L.xlsx", engine='xlsxwriter')
        data.to_excel(writer, sheet_name='Length', index = False)
        writer.save()
        project = {}
        project["filename"] = self.file_name
        project["Space_lines"] = []
        for i,L in enumerate(self.v_lines):
            out = {}
            out["x1"] = L.Lines[0].line().x1()
            out["x2"] = L.Lines[1].line().x1()
            out["Place"] = L.Widget
            out["Role"] = L.LineType
            out["Name"] = self.line_table.item(i,0).text()
            project["Space_lines"].append(out)
        project["Time_lines"] = []
        for i,L in enumerate(self.h_lines):
            out = {}
            out["t"] = L.Lines[0].line().y1()
            out["Name"] = self.time_table.item(i,0).text()
            project["Time_lines"].append(out)
        project["Outliers"] = {
            "Min length": self.MinimumOutlierSpinBox.value(),
            "Max length": self.MaximumOutlierSpinBox.value(),
            "Window size": self.WindowSizeOutlierSpinBox.value()
            }
        with open(f"{self.path}\{self.image_name}{idx_project or ''}.casar", "w") as outfile:
            json.dump(project, outfile)
        self.MessageWindow = QMessageBox()
        self.MessageWindow.setWindowTitle("Status")
        self.MessageWindow.setText("Project was saved successfully")
        self.MessageWindow.exec_()


    def GetStartandEnd(self):
        if self.StartComboBox.currentIndex() == 0:
            t_start = 0
        else:
            t_start = self.h_lines[self.StartComboBox.currentIndex() - 1].Lines[0].line().y1()
        if self.EndComboBox.currentIndex() == 0:
            t_end = self.fluo_image.shape[0]
        else:
            t_end = self.h_lines[self.EndComboBox.currentIndex() - 1].Lines[0].line().y1()
        return int(t_start  + 0.5), int(t_end + 0.5)

    def GetSarcROI(self):
        x_cond_calc_roi = np.zeros(self.trans_image.shape[1], dtype="uint8")
        x_cond_calc_excl_roi = np.ones(self.trans_image.shape[1], dtype="uint8")
        num_lines = 0
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.ActiveSignal and L.Widget == PlacedWidget.Sarcomere:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                x_cond_calc_roi[idx1:idx2] = num_lines + 1
                num_lines += 1
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.EdgeorArtifact and L.Widget == PlacedWidget.Sarcomere:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                x_cond_calc_excl_roi[idx1:idx2] = 0
        return x_cond_calc_roi * x_cond_calc_excl_roi, num_lines

    def ComputeSarProfile(self):
        idx_start, idx_end = self.GetStartandEnd()
        x_cond_calc_roi, n_lines = self.GetSarcROI()
        sar_roi = self.trans_image[idx_start:idx_end]
        N = sar_roi.shape[1]
        l_max = self.MaxLength.value()
        l_min = self.MinLength.value()
        lambda_max = 1 / l_max
        lambda_min = 1 / l_min
        dx = self.dx
        args_list = [(i, sar_roi[i - idx_start],x_cond_calc_roi, n_lines, dx, lambda_max, lambda_min, l_max, l_min) 
                     for i in range(idx_start, idx_end)]
        L = []
        with multiprocessing.Pool(self.ParallelSpinBox.value()) as pool:
            L = pool.map(process_element, args_list)
        self.L = np.array(L) 
        self.t_L = np.arange(idx_start*self.dt, idx_end*self.dt, self.dt)
        self.SarPlot.clear()
        self.outlier_plot = None
        self.SarPlot.plot(self.t_L, self.L)
        self.SarPlot.autoRange()


    def ShowSarcoLineScan(self):
        if self.FFTComboBox.count() == 0:
            error_dialog = QErrorMessage()
            error_dialog.showMessage("Set horizontal line first!")
            error_dialog.exec_()
            return
        idx_line = int(self.h_lines[self.FFTComboBox.currentIndex()].Lines[0].line().y1() + 0.5)
        # get ROI
        data_plot = None
        self.LineScanPlot.clear()
        if self.FFTWidgetComboBox.currentIndex() == 0:
            l_f = self.fluo_image[idx_line]
            x_f = np.arange(0,l_f.shape[0]) * self.dx * 1e-3
            self.LineScanPlot.plot(x_f,l_f)
        else:
            x_cond_calc_roi, n_lines = self.GetSarcROI()
            if np.sum(x_cond_calc_roi) < 1e-5:
                data_plot = self.trans_image[idx_line]
                self.LineScanPlot.plot(x_f,l_f)
            else:
                self.LineScanPlot.addLegend()
                idx_s = 0
                for i in range(n_lines):
                    data_plot = self.trans_image[idx_line, x_cond_calc_roi == i + 1]
                    N = len(data_plot)
                    l_max = self.MaxLength.value()
                    l_min = self.MinLength.value()
                    lambda_max = 1 / l_max
                    lambda_min = 1 / l_min
                    window = signal.windows.hann(N)
                    y_f = fft.fft(data_plot * window / np.sum(window))
                    T = self.dx * 1e-3
                    x_f = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
                    l_f = np.abs(y_f[:N // 2])
                    X_f = x_f[(x_f >= lambda_max) & (x_f <= lambda_min)]
                    L_f = np.abs(y_f[:N // 2])[(x_f >= lambda_max) & (x_f <= lambda_min)]
                    idx_peak = np.argmax(L_f)
                    lm1 = X_f[idx_peak - 1]
                    l0 = X_f[idx_peak]
                    if 0 < idx_peak < len(L_f) - 1:
                        lp1 = X_f[idx_peak + 1]
                    else:
                        lp1 = 1 / X_f[idx_peak]
                    for j, L in enumerate(self.v_lines):
                        if L.LineType == LineType.ActiveSignal and L.Widget == PlacedWidget.Sarcomere:
                            if idx_s == i:
                                pen = pg.mkPen(color=L.Lines[0].pen().color())
                                self.LineScanPlot.plot(x_f,l_f,pen=pen,name=self.line_table.item(j,0).text())
                                break
                            else:
                                idx_s += 1
                pen = QPen(QtCore.Qt.white, 2e-2)
                pen.setStyle(QtCore.Qt.CustomDashLine)
                pen.setDashPattern([5, 50]) 
                line1 = pg.InfiniteLine(angle=90, movable=False)
                self.LineScanPlot.addItem(line1, ignoreBounds=True)
                line1.setPen(pen)
                line1.setPos(lambda_min)
                line2 = pg.InfiniteLine(angle=90, movable=False)
                self.LineScanPlot.addItem(line2, ignoreBounds=True)
                line2.setPen(pen)
                line2.setPos(lambda_max)
        self.LineScanPlot.autoRange()

    def GetFluoROI(self):
        x_cond_calc_roi = np.zeros(self.fluo_image.shape[1],dtype="uint8")
        x_cond_calc_excl_roi = np.ones(self.fluo_image.shape[1],dtype="uint8")
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.ActiveSignal and L.Widget == PlacedWidget.Fluo:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                x_cond_calc_roi[idx1:idx2] = 1
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.EdgeorArtifact and L.Widget == PlacedWidget.Fluo:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                x_cond_calc_excl_roi[idx1:idx2] = 0
        return x_cond_calc_roi * x_cond_calc_excl_roi

    def ComputeFluoProfile(self):
        idx_start, idx_end = self.GetStartandEnd()
        # get background
        x_cond_bg = np.zeros(self.fluo_image.shape[1],dtype="uint8")
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.Background and L.Widget == PlacedWidget.Fluo:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                x_cond_bg[idx1:idx2] = 1
        bg_roi = self.fluo_image[idx_start:idx_end,x_cond_bg == 1]
        if bg_roi.size > 0:
            self.bg_mean = np.mean(bg_roi)
            self.bg_std = np.std(bg_roi)
        else:
            self.bg_mean = 0
            self.bg_std = 0
        img = deepcopy(self.fluo_image)
        img = img - self.bg_mean
        #get ROI
        x_cond_calc_roi = self.GetFluoROI()
        fluo_roi = img[idx_start:idx_end,x_cond_calc_roi == 1]
        if self.MeanCheckBox.isChecked():
            self.F = np.mean(fluo_roi, axis = 1)
        else:
            self.F = np.sum(fluo_roi, axis = 1)
        self.t_F = np.arange(idx_start*self.dt,idx_end*self.dt,self.dt)
        self.LinePlot.clear()
        self.LinePlot.plot(self.t_F, self.F)
        self.LinePlot.autoRange()

    def ComputeFluoArtifacts(self):
        self.ArtifactsPlot.clear()
        self.ArtifactsPlot.addLegend()
        idx_start, idx_end = self.GetStartandEnd()
        self.t_F = np.arange(idx_start*self.dt,idx_end*self.dt,self.dt)
        for i, L in enumerate(self.v_lines):
            if L.LineType == LineType.ActiveSignal and L.Widget == PlacedWidget.Fluo:
                idx1 = int(L.Lines[0].line().x1() + 0.5)
                idx2 = int(L.Lines[1].line().x1() + 0.5)
                pen = pg.mkPen(color=L.Lines[0].pen().color())
                self.ArtifactsPlot.plot(self.t_F,np.mean(self.fluo_image[idx_start:idx_end,idx1:idx2],axis=1),pen=pen,name=self.line_table.item(i,0).text())
        self.ArtifactsPlot.autoRange()

    def ChangeTimeComboBox(self,row,col):
        if col == 0:
            self.StartComboBox.setItemText(row + 1, self.time_table.item(row,col).text())
            self.EndComboBox.setItemText(row + 1, self.time_table.item(row,col).text())
            self.FFTComboBox.setItemText(row, self.time_table.item(row,col).text())

    def Changedt(self,value):
        self.dt = value        
        
    def Changedx(self,value):
        self.dx = value 
        
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Control and self.file_name != "":
            idx = self.img_viewer.currentIndex()
            self.img_viewer.widget(idx).scene.ctrl_pressed = True
    
    def keyReleaseEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Control and self.file_name != "":
            idx = self.img_viewer.currentIndex()
            self.img_viewer.widget(idx).scene.ctrl_pressed = False
                
    def RunColorDialogV(self,row,col):
        if col == 1:
            color_picked = QColorDialog.getColor()
            if not color_picked.isValid():
                return
            pen = self.v_lines[row].Lines[0].pen()
            pen.setColor(color_picked)
            for j in range(2):
                self.v_lines[row].Lines[0].setPen(pen)
                self.v_lines[row].Lines[1].setPen(pen)
            self.line_table.item(row, col).setBackground(color_picked)
            self.line_table.clearSelection()
        
    def RunColorDialogH(self,row,col):
        if col == 1:
            color_picked = QColorDialog.getColor()
            if not color_picked.isValid():
                return
            pen = self.h_lines[row].Lines[0].pen()
            pen.setColor(color_picked)
            for i in range(2):
                self.h_lines[row].Lines[i].setPen(pen)
            self.time_table.item(row, col).setBackground(color_picked)
            self.time_table.clearSelection()
    
    def FindWidget(self,table,idx,widget):
        for i in range(table.rowCount()):
            if table.cellWidget(i,idx) == widget:
                return i
    
    def ChangePositionofHLine(self,value):
        line_number = self.FindWidget(self.time_table,2,self.sender())
        line1 = self.h_lines[line_number].Lines[0]
        line2 = self.h_lines[line_number].Lines[1]
        self.ChangeSpinBox(line1,line1.line().x1(),value/(self.dt*1e-3))
        self.ChangeSpinBox(line2,line2.line().x1(),value/(self.dt*1e-3))
        
    def ChangePositionof1stVLine(self,value):
        line_number = self.FindWidget(self.line_table,2,self.sender())
        line1 = self.v_lines[line_number].Lines[0]
        self.ChangeSpinBox(line1,value/(self.dx*1e-3),line1.line().y1())

    def ChangePositionof2ndVLine(self,value):
        line_number = self.FindWidget(self.line_table,3,self.sender())
        line1 = self.v_lines[line_number].Lines[1]
        self.ChangeSpinBox(line1,value/(self.dx*1e-3),line1.line().y1())

    def ChangeHWidth(self,value):
        line_number = self.FindWidget(self.time_table,3,self.sender())
        pen = self.h_lines[line_number].Lines[0].pen()
        pen.setWidth(value)
        for i in range(2):
            self.h_lines[line_number].Lines[i].setPen(pen)
                
    def ChangeVWidth(self,value):
        line_number = self.FindWidget(self.line_table,4,self.sender())
        pen = self.v_lines[line_number].Lines[0].pen()
        pen.setWidth(value)
        for i in range(2):
            self.v_lines[line_number].Lines[i].setPen(pen)
    
    def ChangeVLinesRole(self, value):
        line_number = self.FindWidget(self.line_table,5,self.sender())
        self.v_lines[line_number].LineType = value
    
    def RemoveLine(self,line,widget_idx):
        self.img_viewer.widget(widget_idx).scene.removeItem(line)
    
    def DeleteHLine(self):
        line_number = self.FindWidget(self.time_table,4,self.sender())
        self.time_table.removeRow(line_number)
        lines = self.h_lines.pop(line_number)
        for i in range(2):
            self.RemoveLine(lines.Lines[i],i)
        self.StartComboBox.removeItem(line_number + 1)
        self.EndComboBox.removeItem(line_number + 1)
        self.FFTComboBox.removeItem(line_number)
                
    def DeleteVLines(self):
        line_number = self.FindWidget(self.line_table,7,self.sender())
        self.line_table.removeRow(line_number)
        lines = self.v_lines.pop(line_number)
        for i in range(2):
            self.RemoveLine(lines.Lines[i],lines.Widget)
    
    def addLineScansRow(self):
        rowPosition = self.line_table.rowCount()
        self.line_table.insertRow(rowPosition)
        self.line_table.setItem(rowPosition , 0, QTableWidgetItem(f"Lines #{rowPosition+1}"))
        self.line_table.setItem(rowPosition , 1, QTableWidgetItem(""))
        clr = self.v_lines[-1].Lines[0].pen().color()
        self.line_table.item(rowPosition, 1).setBackground(clr)
        x1_spinbox = QDoubleSpinBox()
        x = self.v_lines[-1].Lines[0].line().x1()
        x1_spinbox.setDecimals(3)
        x1_spinbox.setMinimum(0)
        x1_spinbox.setMaximum(self.fluo_image.shape[1]*self.dx*1e-3)
        x1_spinbox.setValue(x*self.dx*1e-3)
        x1_spinbox.valueChanged.connect(self.ChangePositionof1stVLine)
        self.line_table.setCellWidget(rowPosition , 2, x1_spinbox)
        x2_spinbox = QDoubleSpinBox()
        x = self.v_lines[-1].Lines[1].line().x1()
        x2_spinbox.setDecimals(3)
        x1_spinbox.setMinimum(0)
        x2_spinbox.setMaximum(self.fluo_image.shape[1]*self.dx*1e-3)
        x2_spinbox.setValue(x*self.dx*1e-3)
        x2_spinbox.valueChanged.connect(self.ChangePositionof2ndVLine)
        self.line_table.setCellWidget(rowPosition , 3, x2_spinbox)
        width_spinbox = QSpinBox()
        x = self.v_lines[-1].Lines[0].pen().width()
        width_spinbox.setValue(x)
        width_spinbox.valueChanged.connect(self.ChangeVWidth)
        self.line_table.setCellWidget(rowPosition , 4, width_spinbox)
        role_combobox = QComboBox()
        role_combobox.addItem("Background")
        role_combobox.addItem("Active signal")
        role_combobox.addItem("Edge/Artifact")
        x = self.v_lines[-1].LineType #role
        role_combobox.setCurrentIndex(x.value)
        role_combobox.currentIndexChanged.connect(self.ChangeVLinesRole)
        self.line_table.setCellWidget(rowPosition , 5, role_combobox)
        if self.v_lines[-1].Widget == PlacedWidget.Fluo:
            self.line_table.setItem(rowPosition , 6, QTableWidgetItem("Fluorescence"))
        else:
            self.line_table.setItem(rowPosition , 6, QTableWidgetItem("Sarcomere"))
        cell_item = self.line_table.item(rowPosition, 6)
        cell_item.setFlags(cell_item.flags() ^ QtCore.Qt.ItemIsEditable)
        delete_button = QPushButton("Remove Lines")
        delete_button.clicked.connect(self.DeleteVLines)
        self.line_table.setCellWidget(rowPosition , 7, delete_button)
   
    def addTimesRow(self):
        rowPosition = self.time_table.rowCount()
        self.time_table.insertRow(rowPosition)
        self.time_table.setItem(rowPosition , 0, QTableWidgetItem(f"Line #{rowPosition+1}"))
        self.time_table.setItem(rowPosition , 1, QTableWidgetItem(""))
        clr = self.h_lines[-1].Lines[0].pen().color()
        self.time_table.item(rowPosition, 1).setBackground(clr)
        t_spinbox = QDoubleSpinBox()
        t = self.h_lines[-1].Lines[0].line().y1()
        t_spinbox.setDecimals(3)
        t_spinbox.setMinimum(0)
        t_spinbox.setMaximum(self.fluo_image.shape[0]*self.dt*1e-3)
        t_spinbox.setValue(t*self.dt*1e-3)
        t_spinbox.valueChanged.connect(self.ChangePositionofHLine)
        self.time_table.setCellWidget(rowPosition , 2, t_spinbox)
        width_spinbox = QSpinBox()
        x = self.h_lines[-1].Lines[0].pen().width()
        width_spinbox.setValue(x)
        width_spinbox.valueChanged.connect(self.ChangeHWidth)
        self.time_table.setCellWidget(rowPosition , 3, width_spinbox)
        delete_button = QPushButton("Remove Lines")
        delete_button.clicked.connect(self.DeleteHLine)
        self.time_table.setCellWidget(rowPosition , 4, delete_button)
   
    def NewHorizontalLine(self,idx,line):
        lines = Line()
        lines.Lines[idx] = line
        idx2 = (idx+1) % 2
        lines.Lines[idx2] = self.img_viewer.widget(idx2).scene.externalAddHLine(line)
        lines.LineType = LineType.TimeSegment
        lines.Widget = PlacedWidget.Both
        self.h_lines.append(lines)
        self.addTimesRow()
        rowPosition = self.time_table.rowCount()
        self.StartComboBox.addItem(f"Line #{rowPosition}")
        self.FFTComboBox.addItem(f"Line #{rowPosition}")
        self.EndComboBox.addItem(f"Line #{rowPosition}")
        
    def NewHorizontalLine1(self,line):
        return self.NewHorizontalLine(0,line)
        
    def NewHorizontalLine2(self,line):
        return self.NewHorizontalLine(1,line)
    
    def NewVerticalLines(self,idx,line,line2):
        lines = Line()
        if line.line().x1() < line2.line().x1():
            lines.Lines[0] = line
            lines.Lines[1] = line2
        else:
            lines.Lines[0] = line2
            lines.Lines[1] = line
        if idx == 0:
            lines.LineType = LineType.Background
        else:
            lines.LineType = LineType.ActiveSignal
        lines.Widget = idx
        self.v_lines.append(lines)
        self.addLineScansRow()
        
    def NewVerticalLines1(self,line,line2):
        return self.NewVerticalLines(0,line,line2)
        
    def NewVerticalLines2(self,line,line2):
        return self.NewVerticalLines(1,line,line2)
    
    def FindLine(self,line): #find line among the existing lines returns index in table,widget index and index of line in case of vertical
        for i, elem in enumerate(self.h_lines):
            for j in range(2):
                if elem.Lines[j] == line:
                    return i,j,-1
        for i, elem in enumerate(self.v_lines):
            for j in range(2):
                if elem.Lines[0] == line:
                        return i,j,0
                if elem.Lines[1] == line:
                        return i,j,1

    def ChangeSpinBox(self,line,x,y):
        idxt,idxw,idxl = self.FindLine(line) #index of line in column, widget (1 or 2), and index of line in case of vertical line (or -1)
        if idxl == -1:
            line2 = self.h_lines[idxt].Lines[idxw]
            if not isinstance(self.sender(), QDoubleSpinBox):
                self.time_table.cellWidget(idxt , 2).setValue(self.h_lines[idxt].Lines[idxw].line().y1()*self.dt*1e-3)
            idxw2 = (idxw + 1) % 2
            self.img_viewer.widget(idxw2).scene.MoveLine(line2, x, y)
        else:
            if not isinstance(self.sender(), QDoubleSpinBox):
                self.line_table.cellWidget(idxt , 2 + idxl).setValue(self.v_lines[idxt].Lines[idxl].line().x1()*self.dx*1e-3)
            else:
                self.img_viewer.widget(idxw).scene.MoveLine(self.v_lines[idxt].Lines[idxl], x, y)

    def ChangeZoom(self,idx):
        self.change_y_only = not self.change_y_only
        if self.change_y_only:
            self.img_viewer.widget((idx+1) % 2).zoom_Y.setChecked(not self.img_viewer.widget((idx+1) % 2).zoom_Y.isChecked())
        
    def ChangeZoom1(self):
        self.ChangeZoom(0)
        
    def ChangeZoom2(self):
        self.ChangeZoom(1)
    
    def SyncChannelsViewers(self):
        #vertical scrollbars
        self.img_viewer.widget(0).img.verticalScrollBar().valueChanged.connect(self.img_viewer.widget(1).img.verticalScrollBar().setValue)
        self.img_viewer.widget(1).img.verticalScrollBar().valueChanged.connect(self.img_viewer.widget(0).img.verticalScrollBar().setValue)
        #horizontal scrollbars
        self.img_viewer.widget(0).img.horizontalScrollBar().valueChanged.connect(self.img_viewer.widget(1).img.horizontalScrollBar().setValue)
        self.img_viewer.widget(1).img.horizontalScrollBar().valueChanged.connect(self.img_viewer.widget(0).img.horizontalScrollBar().setValue)
        self.img_viewer.widget(0).zoom_Y.stateChanged.connect(self.ChangeZoom1)
        self.img_viewer.widget(1).zoom_Y.stateChanged.connect(self.ChangeZoom2)
        #new lines
        self.img_viewer.widget(0).scene.new_horizontal_line_created.connect(self.NewHorizontalLine1)
        self.img_viewer.widget(1).scene.new_horizontal_line_created.connect(self.NewHorizontalLine2)
        self.img_viewer.widget(0).scene.new_vertical_lines_created.connect(self.NewVerticalLines1)
        self.img_viewer.widget(1).scene.new_vertical_lines_created.connect(self.NewVerticalLines2)
        #change line position
        self.img_viewer.widget(0).scene.line_is_moved.connect(self.ChangeSpinBox)
        self.img_viewer.widget(1).scene.line_is_moved.connect(self.ChangeSpinBox)
        #change image position (zoom and x-y)
        self.img_viewer.widget(0).scene.image_position_changed.connect(self.img_viewer.widget(1).scene.changeImgPosition)
        self.img_viewer.widget(1).scene.image_position_changed.connect(self.img_viewer.widget(0).scene.changeImgPosition)
        self.img_viewer.widget(0).sync_zoom_reset.connect(self.img_viewer.widget(1).ZoomResetExt)
        self.img_viewer.widget(1).sync_zoom_reset.connect(self.img_viewer.widget(0).ZoomResetExt)
        self.img_viewer.widget(0).sync_scale_x.connect(self.img_viewer.widget(1).ChangeScaleXExt)
        self.img_viewer.widget(1).sync_scale_x.connect(self.img_viewer.widget(0).ChangeScaleXExt)
        self.img_viewer.widget(0).sync_scale_y.connect(self.img_viewer.widget(1).ChangeScaleYExt)
        self.img_viewer.widget(1).sync_scale_y.connect(self.img_viewer.widget(0).ChangeScaleYExt)
        self.img_viewer.widget(0).sync_change_mode.connect(self.img_viewer.widget(1).ChangeModeExt)
        self.img_viewer.widget(1).sync_change_mode.connect(self.img_viewer.widget(0).ChangeModeExt)
        
    def SetInfo(self,x,y):
        Dx = x*self.dx*1e-3
        Dt = y*self.dt*1e-3
        self.coord_label.setText(f"{Dx:.3f} um, {Dt:.3f} s")
        
    def SetImageViewer(self):
        self.img_viewer.clear()
        lut = [qRgb(0,i,0) for i in range(256)]
        self.img_viewer.addTab(ImageViewer(self.fluo_image,False,lut,self),"Fluo")
        self.fluo_image = self.fluo_image.astype("float64")
        self.fluo_image /= 255.0
        self.img_viewer.widget(0).img.mouse_coord.connect(self.SetInfo)
        self.img_viewer.addTab(ImageViewer(self.trans_image,True,lut,self),"Trans")
        self.img_viewer.widget(1).img.mouse_coord.connect(self.SetInfo)
        self.SyncChannelsViewers()
        
    #Slot for opening the images
    def OpenImage(self):
        filename = QFileDialog.getOpenFileName(self,"Open Image",self.path,"OME-TIFF files (*.tif)") [0]
        if filename != "":
            self.path, self.file_name = os.path.split(os.path.abspath(filename))
            tif = tifffile.TiffFile(filename)
            if len(tif.pages) != 2:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Wrong Image!")
                msg.setInformativeText('File must contain exactly 2 pages(channels): fluorescence and transmission.')
                msg.setWindowTitle("Error!")
                msg.exec_()
                return
            self.ClearAll()
            self.image_name,ext = os.path.splitext(self.file_name)
            while ext:
                self.image_name, ext = os.path.splitext(self.image_name)
            self.fluo_image = tif.pages[0].asarray()
            self.trans_image = tif.pages[1].asarray()
            metadata = tif.ome_metadata
            tree = ET.fromstring(metadata)
            self.dx = float(tree.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')[0].attrib["PhysicalSizeX"]) # 'cause usually in micrometers, conversion um -> nm
            self.dt = float(tree.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')[0].attrib["PhysicalSizeY"]) # 'cause usually in seconds, conversion s -> ms
            self.dt_spinbox.setValue(self.dt)
            self.dx_spinbox.setValue(self.dx)
            self.status.setText("Current Image: " + filename)
            self.time_table.setRowCount(0)
            self.line_table.setRowCount(0)
            self.time_points = ["Start","End"]
            self.SetImageViewer()
            
def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


