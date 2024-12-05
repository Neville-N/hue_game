import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent

class Image(QLabel):
    def __init__(self, parent: QWidget, image: np.ndarray) -> None:
        super().__init__(parent)
        self.pixel_width, self.pixel_height, _ = np.shape(image)
        q_image = QImage(image, self.pixel_width, self.pixel_height, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap(q_image))

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        print(ev.pos())
        return super().mousePressEvent(ev)

class Interface(QApplication):
    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.window = Window()

    def start(self) -> None:
        self.window.show()
        sys.exit(self.exec_())


class Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        numpy_image = np.full((500, 500, 3), 100, dtype=np.uint8)
        self.image = Image(self, numpy_image)
        self.setFixedSize(self.image.pixel_width, self.image.pixel_height)