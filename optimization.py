import scipy
import numpy as np
from ownFuncs.shape import Shape
from ownFuncs.shapes import Shapes
import matplotlib.pyplot as plt
import cv2
import ownFuncs.funcs as of
from matplotlib import use as matplotlib_use
matplotlib_use('TkAgg')


def TransformFunc(C: np.ndarray, XX: np.ndarray, YY: np.ndarray, order: int) -> np.ndarray:
    """Transforms XY coordinates to color values according to transform matrix C

    Args:
        C (np.ndarray): Transform matrix
        XX (np.ndarray): X coordinates
        YY (np.ndarray): Y coordinates

    Returns:
        np.ndarray: Color values
    """
    if order == 1:
        return np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(XX.shape)
    else:
        return np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C)


def findC(data: np.ndarray, order: int) -> np.ndarray:
    """Finds the matrix C that can be used to convert a XY coordinate to a Color value

    Args:
        data (np.ndarray): Sample points 

    Returns:
        np.ndarray: C matrix
    """
    if order == 1:
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    else:
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(
            data[:, :2], axis=1), data[:, :2]**2]

    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    return C


def getMeshGrids(datas, Cs, order: int):
    # regular grid covering the domain of the data
    N = 16
    data = datas[0]
    X = data[:, 0]
    Y = data[:, 1]
    X_MG, Y_MG = np.meshgrid(np.linspace(min(X)-100, max(X)+100, N),
                             np.linspace(min(Y)-100, max(Y)+100, N))

    XX = X_MG.flatten()
    YY = Y_MG.flatten()

    Zs = [TransformFunc(c, XX, YY, order) for c in Cs]
    Z_MGs = np.array([Z.reshape(X_MG.shape) for Z in Zs])
    return X_MG, Y_MG, Z_MGs


def plotSurfaces(datas, MGs, datas2=None) -> None:
    colors = ['blue', 'green', 'red']
    X_MG = MGs[0]
    Y_MG = MGs[1]
    Z_MGs = MGs[2]

    fig = plt.figure(figsize=(20, 20))
    for i in range(3):
        data = datas[i]
        Z_MG = Z_MGs[i]

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(X_MG, Y_MG, Z_MG, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i], s=50)
        if datas2 is not None:
            data2 = datas2[i]
            ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='k', s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(colors[i])
        ax.axis('equal')
        ax.axis('tight')
        ax.set_zlim((0, 255))


def fitSurface(shapes: Shapes, order: int, useLocked: bool = True) -> None:
    if useLocked:
        shapeList = shapes.locked
    else:
        shapeList = shapes.all

    XYR = np.array([[s.centerX, s.centerY, s.color[2]] for s in shapeList])
    XYG = np.array([[s.centerX, s.centerY, s.color[1]] for s in shapeList])
    XYB = np.array([[s.centerX, s.centerY, s.color[0]] for s in shapeList])

    datas = [XYB, XYG, XYR]
    Cs = [findC(d, order) for d in datas]
    MGs = getMeshGrids(datas, Cs, order)
    # plotSurfaces(datas, MGs)
    return datas, MGs, Cs


def setShapeColorEstimations(shapes: Shapes, Cs: np.ndarray, order: int):
    for shape in shapes.all:
        shape.colorEst = np.array([TransformFunc(C, np.array([shape.centerX]),
                                                 np.array([shape.centerY]), order) for C in Cs]).flatten().astype(np.uint8)


Npuzzle = '15'
src = f'data/hue_scrambled_{Npuzzle}.png'
# src = f'data/hue_solved_{Npuzzle}.png'
img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)

shapes = Shapes(img, Npuzzle)
order = 1
datas, MGs, Cs = fitSurface(shapes, order)
setShapeColorEstimations(shapes, Cs, order)

XYB = []
XYG = []
XYR = []
stepcount = -1

while len(shapes.unlocked) > 0:
    # shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)
    # shapes.unlocked.sort(key=lambda s: s.neighboursDistance(), reverse=True)
    shapes.unlocked.sort(key=lambda s: s.distToEstimation, reverse=True)
    shape = shapes.unlocked[0]
    stepcount += 1
    BGR = np.array([TransformFunc(C, np.array([shape.centerX]),
                                  np.array([shape.centerY]), order) for C in Cs]).flatten().astype(np.uint8)
    closestShape = shapes.findShapeClosestToColor(BGR, shape)
    # print(f"Old color: {shape.colorA}")
    # print(f"New color: {BGR}")
    # print(f"closest c: {closestShape.colorA}\n")
    shapes.swapShapes(shape, closestShape)
    if not np.all(np.equal(shape.colorA, closestShape.colorA)):
        shapes.markSwappedShapes(shape, closestShape)
        of.saveImg(
            shapes.img, f"data/solveanimation{Npuzzle}/", f"step_{stepcount}.png")

    XYB.append([shape.centerX, shape.centerY, shape.color[0]])
    XYG.append([shape.centerX, shape.centerY, shape.color[1]])
    XYR.append([shape.centerX, shape.centerY, shape.color[2]])

datas2 = np.array([XYB, XYG, XYR])
plotSurfaces(datas, MGs, datas2)


order = 2
somethingChanged = True
limit = 0
while somethingChanged and limit < 20:
    limit += 1
    somethingChanged = False
    _, MGs, Cs = fitSurface(shapes, order)
    shapes.resetLocks()
    setShapeColorEstimations(shapes, Cs, order)

    XYB = []
    XYG = []
    XYR = []
    while len(shapes.unlocked) > 0:
        # shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)
        # shapes.unlocked.sort(key=lambda s: s.neighboursDistance(), reverse=True)
        shapes.unlocked.sort(key=lambda s: s.distToEstimation, reverse=True)
        shape = shapes.unlocked[0]
        stepcount += 1
        BGR = np.array([TransformFunc(C, np.array([shape.centerX]),
                                      np.array([shape.centerY]), order) for C in Cs]).flatten().astype(np.uint8)
        closestShape = shapes.findShapeClosestToColor(BGR, shape)
        # print(f"Old color         : {shape.colorA}")
        # print(f"Surface est. color: {BGR}")
        # print(f"closest color     : {closestShape.colorA}\n")
        shapes.swapShapes(shape, closestShape)
        shapes.markSwappedShapes(shape, closestShape)
        if not np.all(np.equal(shape.colorA, closestShape.colorA)):
            somethingChanged = True
            of.saveImg(
                shapes.img, f"data/solveanimation{Npuzzle}/", f"step_{stepcount}.png")

        XYB.append([shape.centerX, shape.centerY, shape.color[0]])
        XYG.append([shape.centerX, shape.centerY, shape.color[1]])
        XYR.append([shape.centerX, shape.centerY, shape.color[2]])
    datas2 = np.array([XYB, XYG, XYR])
    plotSurfaces(datas, MGs, datas2)

of.saveImg(
    shapes.img, f"data/solveanimation{Npuzzle}/", f"step_{stepcount}.png")

print(f"P{Npuzzle} Ran order two loop for {limit} times")
# datas2 = np.array([XYB, XYG, XYR])
# plotSurfaces(datas, MGs, datas2)

plt.show()
