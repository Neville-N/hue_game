import scipy
import numpy as np
from ownFuncs.shape import Shape
from ownFuncs.shapes import Shapes
import matplotlib.pyplot as plt
import cv2
import ownFuncs.funcs as of
from matplotlib import use as matplotlib_use
matplotlib_use('TkAgg')


def findSurface(shapes: Shapes, useLocked: bool = True):
    if useLocked:
        shapeList = shapes.locked
    else:
        shapeList = shapes.all

    X = np.array([s.centerX for s in shapeList])
    Y = np.array([s.centerY for s in shapeList])
    # R = np.array([s.color[2] for s in shapeList])
    # G = np.array([s.color[1] for s in shapeList])
    # B = np.array([s.color[0] for s in shapeList])

    XYR = np.array([[s.centerX, s.centerY, s.color[2]] for s in shapeList])
    XYG = np.array([[s.centerX, s.centerY, s.color[1]] for s in shapeList])
    XYB = np.array([[s.centerX, s.centerY, s.color[0]] for s in shapeList])

    # regular grid covering the domain of the data
    N = 16
    Xr, Yr = np.meshgrid(np.linspace(np.min(X), max(X), N),
                         np.linspace(np.min(Y), max(Y), N))
    XX = Xr.flatten()
    YY = Yr.flatten()

    def findZs(data):
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(
            data[:, :2], axis=1), data[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        return np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C).reshape(Xr.shape)
    
    datas = [XYR, XYG, XYB]
    colors = ['red', 'green', 'blue']
    fig = plt.figure()

    for i in range(3):
        data = datas[i]
        Z = findZs(data)
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(Xr, Yr, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i], s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(colors[i])
        ax.axis('equal')
        ax.axis('tight')

    plt.show()


Npuzzle = '2'
# src = f'data/hue_scrambled_{Npuzzle}.png'
src = f'data/hue_solved_{Npuzzle}.png'
img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)

shapes = Shapes(img, Npuzzle)

findSurface(shapes)
