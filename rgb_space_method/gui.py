import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
)
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from colors import Colors

SIZE = 100


class Image(QLabel):
    def __init__(self, color: np.ndarray, size) -> None:
        super().__init__()
        image = np.tile(color * 255, (size, size))
        q_image = QImage(
            np.ndarray.astype(image, np.uint8), size, size, QImage.Format.Format_RGB888
        )
        self.setPixmap(QPixmap(q_image))


class Interface(QApplication):
    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.window = Window()

    def start(self) -> None:
        self.window.show()
        sys.exit(self.exec_())


class Solver(QPushButton):
    def __init__(self, callback: callable) -> None:
        super().__init__("Solve")
        self.clicked.connect(callback)


class Window(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.central_layout = QGridLayout(self)

        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)


        self.colors = Colors()
        self.grid = Grid(self.colors)
        self.plot = Plot(self.colors)

        self.central_layout.addWidget(self.grid, 0, 0)
        self.central_layout.addWidget(self.plot, 0, 1)
        self.central_layout.addWidget(self.solve_button, 1, 0)
        self.central_layout.addWidget(self.reset_button, 1, 1)

    def update(self):
        plot_state = self.plot.state
        self.central_layout.removeWidget(self.grid)
        self.central_layout.removeWidget(self.plot)
        self.grid = Grid(self.colors)
        self.plot = Plot(self.colors)
        self.plot.set_state(plot_state)
        self.central_layout.addWidget(self.grid, 0, 0)
        self.central_layout.addWidget(self.plot, 0, 1)
    
    def solve(self):
        self.colors.solve()
        self.update()
    
    def reset(self):
        self.colors = Colors()
        self.update()

class Grid(QWidget):
    def __init__(self, colors: Colors) -> None:
        super().__init__()

        layout = QGridLayout(self)
        for row in range(colors.size):
            for column in range(colors.size):
                image = Image(colors.color(row, column), SIZE)
                layout.addWidget(image, row, column)


class Plot(QWidget):
    def __init__(self, colors: Colors) -> None:
        super().__init__()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.add_subplot(111, projection="3d")

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)

        for i in range(len(colors.grid)):
            for j in range(len(colors.grid)):
                home = colors.color(i, j)
                self.axes.scatter(*home, color=home)
                neighbours = colors.neihgbours(i, j)
                for neighbour in neighbours:
                    self.axes.plot(
                        [home[0], neighbour[0]],
                        [home[1], neighbour[1]],
                        [home[2], neighbour[2]],
                        c="k",
                    )

    def set_state(self, state: list):
        self.axes.azim = state[0]
        self.axes.elev = state[1]
        self.axes.dist = state[2]

    @property
    def state(self):
        state = []
        state.append(self.axes.azim)
        state.append(self.axes.elev)
        state.append(self.axes.dist)
        return state
