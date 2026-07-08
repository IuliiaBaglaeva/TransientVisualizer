import os
import sys

import numpy as np
import tifffile
from PyQt5 import QtCore
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage


class CropPreviewLabel(QLabel):
    cropSelected = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(900, 700)
        self.image_size = None
        self.display_rect = QRect()
        self.dragging = False
        self.drag_start = None
        self.drag_current = None

    def setPreviewPixmap(self, pixmap, image_size):
        self.image_size = image_size
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self.display_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def widgetToImage(self, point):
        if self.image_size is None or self.display_rect.width() <= 0 or self.display_rect.height() <= 0:
            return None
        if not self.display_rect.contains(point):
            return None
        rel_x = (point.x() - self.display_rect.x()) / self.display_rect.width()
        rel_y = (point.y() - self.display_rect.y()) / self.display_rect.height()
        image_w, image_h = self.image_size
        x = int(rel_x * image_w)
        y = int(rel_y * image_h)
        x = min(max(x, 0), image_w - 1)
        y = min(max(y, 0), image_h - 1)
        return QPoint(x, y)

    def buildSquareCrop(self, start, end):
        image_w, image_h = self.image_size
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        size = max(abs(dx), abs(dy)) + 1
        size = min(size, image_w, image_h)
        x = start.x() if dx >= 0 else start.x() - size + 1
        y = start.y() if dy >= 0 else start.y() - size + 1
        x = min(max(x, 0), image_w - size)
        y = min(max(y, 0), image_h - size)
        return x, y, size

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and QApplication.keyboardModifiers() & Qt.ControlModifier:
            image_point = self.widgetToImage(event.pos())
            if image_point is not None:
                self.dragging = True
                self.drag_start = image_point
                self.drag_current = image_point
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            image_point = self.widgetToImage(event.pos())
            if image_point is not None:
                self.drag_current = image_point
                if self.drag_start is not None:
                    x, y, size = self.buildSquareCrop(self.drag_start, self.drag_current)
                    self.cropSelected.emit(x, y, size)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging and event.button() == Qt.LeftButton:
            image_point = self.widgetToImage(event.pos())
            if image_point is not None:
                self.drag_current = image_point
            if self.drag_start is not None and self.drag_current is not None:
                x, y, size = self.buildSquareCrop(self.drag_start, self.drag_current)
                self.cropSelected.emit(x, y, size)
            self.dragging = False
            self.drag_start = None
            self.drag_current = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SquareCropper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Square TIFF Cropper")
        self.resize(1200, 900)

        self.file_path = ""
        self.fluo_image = None
        self.trans_image = None
        self.dx = None
        self.dt = None
        self.rotated_cache = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout()
        central.setLayout(root)

        controls = QHBoxLayout()
        root.addLayout(controls)

        self.open_button = QPushButton("Open TIFF")
        self.open_button.clicked.connect(self.open_image)
        controls.addWidget(self.open_button)

        self.save_button = QPushButton("Save Cropped TIFF")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        controls.addWidget(self.save_button)

        controls.addWidget(QLabel("Channel"))
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Fluo", "Trans"])
        self.channel_box.currentIndexChanged.connect(self.update_preview)
        controls.addWidget(self.channel_box)

        controls.addWidget(QLabel("Angle"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(0.5)
        self.angle_spin.setSuffix(" deg")
        self.angle_spin.valueChanged.connect(self.on_transform_changed)
        controls.addWidget(self.angle_spin)

        self.center_button = QPushButton("Center Crop")
        self.center_button.clicked.connect(self.center_crop)
        self.center_button.setEnabled(False)
        controls.addWidget(self.center_button)

        controls.addStretch(1)

        crop_grid = QGridLayout()
        root.addLayout(crop_grid)

        crop_grid.addWidget(QLabel("X"), 0, 0)
        self.crop_x = QSpinBox()
        self.crop_x.valueChanged.connect(self.update_preview)
        crop_grid.addWidget(self.crop_x, 0, 1)

        crop_grid.addWidget(QLabel("Y"), 0, 2)
        self.crop_y = QSpinBox()
        self.crop_y.valueChanged.connect(self.update_preview)
        crop_grid.addWidget(self.crop_y, 0, 3)

        crop_grid.addWidget(QLabel("Size"), 0, 4)
        self.crop_size = QSpinBox()
        self.crop_size.setMinimum(1)
        self.crop_size.valueChanged.connect(self.on_crop_size_changed)
        crop_grid.addWidget(self.crop_size, 0, 5)

        self.preview_label = CropPreviewLabel()
        self.preview_label.setText("Open a two-page OME-TIFF to begin")
        self.preview_label.cropSelected.connect(self.apply_crop_selection)
        root.addWidget(self.preview_label, 1)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

    def open_image(self):
        filename = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "OME-TIFF files (*.tif *.tiff)",
        )[0]
        if not filename:
            return

        tif = tifffile.TiffFile(filename)
        if len(tif.pages) != 2:
            QMessageBox.critical(
                self,
                "Error",
                "The file must contain exactly 2 pages: fluorescence and transmission.",
            )
            return

        self.file_path = filename
        self.fluo_image = tif.pages[0].asarray()
        self.trans_image = tif.pages[1].asarray()
        self.rotated_cache.clear()

        self.dx = None
        self.dt = None
        if tif.ome_metadata:
            metadata = tifffile.xml2dict(tif.ome_metadata)
            try:
                pixels = metadata["OME"]["Image"]["Pixels"]
                self.dx = float(pixels["PhysicalSizeX"])
                self.dt = float(pixels["PhysicalSizeY"])
            except Exception:
                self.dx = None
                self.dt = None

        if self.fluo_image.shape != self.trans_image.shape:
            QMessageBox.critical(
                self,
                "Error",
                "Both TIFF pages must have the same size.",
            )
            self.file_path = ""
            self.fluo_image = None
            self.trans_image = None
            return

        height, width = self.fluo_image.shape
        square_size = min(height, width)
        self.crop_size.blockSignals(True)
        self.crop_size.setMaximum(square_size)
        self.crop_size.setValue(square_size)
        self.crop_size.blockSignals(False)

        self.crop_x.setMaximum(width - square_size)
        self.crop_y.setMaximum(height - square_size)
        self.crop_x.setValue((width - square_size) // 2)
        self.crop_y.setValue((height - square_size) // 2)

        self.save_button.setEnabled(True)
        self.center_button.setEnabled(True)
        self.status_label.setText(os.path.basename(filename))
        self.update_preview()

    def on_transform_changed(self):
        self.rotated_cache.clear()
        self.update_preview()

    def on_crop_size_changed(self):
        if self.fluo_image is None:
            return
        height, width = self.fluo_image.shape
        size = self.crop_size.value()
        self.crop_x.setMaximum(max(width - size, 0))
        self.crop_y.setMaximum(max(height - size, 0))
        self.crop_x.setValue(min(self.crop_x.value(), self.crop_x.maximum()))
        self.crop_y.setValue(min(self.crop_y.value(), self.crop_y.maximum()))
        self.update_preview()

    def center_crop(self):
        if self.fluo_image is None:
            return
        height, width = self.fluo_image.shape
        size = self.crop_size.value()
        self.crop_x.setValue((width - size) // 2)
        self.crop_y.setValue((height - size) // 2)

    def apply_crop_selection(self, x, y, size):
        self.crop_size.blockSignals(True)
        self.crop_x.blockSignals(True)
        self.crop_y.blockSignals(True)
        self.crop_size.setValue(size)
        self.crop_x.setMaximum(max(self.fluo_image.shape[1] - size, 0))
        self.crop_y.setMaximum(max(self.fluo_image.shape[0] - size, 0))
        self.crop_x.setValue(x)
        self.crop_y.setValue(y)
        self.crop_size.blockSignals(False)
        self.crop_x.blockSignals(False)
        self.crop_y.blockSignals(False)
        self.update_preview()

    def get_rotated_image(self, channel_name):
        angle = round(self.angle_spin.value(), 3)
        cache_key = (channel_name, angle)
        if cache_key in self.rotated_cache:
            return self.rotated_cache[cache_key]

        source = self.fluo_image if channel_name == "fluo" else self.trans_image
        if abs(angle) < 1e-9:
            rotated = source.copy()
        else:
            rotated = ndimage.rotate(source, angle, reshape=False, order=1, mode="nearest")
            rotated = self.cast_like_source(rotated, source.dtype)
        self.rotated_cache[cache_key] = rotated
        return rotated

    def cast_like_source(self, image, dtype):
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            image = np.clip(image, info.min, info.max)
        return image.astype(dtype)

    def get_crop_bounds(self):
        x = self.crop_x.value()
        y = self.crop_y.value()
        size = self.crop_size.value()
        return x, y, size

    def get_cropped_image(self, channel_name):
        rotated = self.get_rotated_image(channel_name)
        x, y, size = self.get_crop_bounds()
        return rotated[y : y + size, x : x + size]

    def normalize_preview(self, image):
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32)
        low = np.percentile(image, 1)
        high = np.percentile(image, 99)
        if high <= low:
            high = low + 1.0
        preview = np.clip((image - low) * 255.0 / (high - low), 0, 255)
        return preview.astype(np.uint8)

    def update_preview(self):
        if self.fluo_image is None:
            return

        channel_name = "fluo" if self.channel_box.currentIndex() == 0 else "trans"
        rotated = self.get_rotated_image(channel_name)
        preview = self.normalize_preview(rotated)
        rgb = np.repeat(preview[:, :, None], 3, axis=2)

        height, width = rgb.shape[:2]
        q_image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(Qt.red, 3))
        x, y, size = self.get_crop_bounds()
        painter.drawRect(x, y, size, size)
        painter.end()

        self.preview_label.setPreviewPixmap(pixmap, (width, height))
        self.status_label.setText(
            f"{os.path.basename(self.file_path)} | Ctrl+drag square crop | angle={self.angle_spin.value():.1f} | "
            f"x={x}, y={y}, size={size}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def save_image(self):
        if self.fluo_image is None:
            return

        out_path = QFileDialog.getSaveFileName(
            self,
            "Save Cropped Image",
            self.default_output_name(),
            "OME-TIFF files (*.tif *.tiff)",
        )[0]
        if not out_path:
            return

        cropped_fluo = self.get_cropped_image("fluo")
        cropped_trans = self.get_cropped_image("trans")
        stack = np.stack([cropped_fluo, cropped_trans], axis=0)

        metadata = {"axes": "CYX"}
        if self.dx is not None:
            metadata["PhysicalSizeX"] = self.dx
        if self.dt is not None:
            metadata["PhysicalSizeY"] = self.dt

        tifffile.imwrite(
            out_path,
            stack,
            ome=True,
            photometric="minisblack",
            metadata=metadata,
        )
        QMessageBox.information(self, "Saved", f"Saved cropped TIFF:\n{out_path}")

    def default_output_name(self):
        if not self.file_path:
            return "cropped_square.tif"
        root, ext = os.path.splitext(self.file_path)
        return f"{root}_square{ext}"


def main():
    app = QApplication(sys.argv)
    window = SquareCropper()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
