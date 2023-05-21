# from matplotlib import use as matplotlib_use
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

import ownFuncs.funcs as of
from ownFuncs.shape import Shape
from ownFuncs.shapes import Shapes


def TransformFunc(C: npt.NDArray, XX: npt.NDArray, YY: npt.NDArray, order: int) -> npt.NDArray:
    """Transforms XY coordinates to color values according to transform matrix C

    Args:
        C (npt.NDArray): Transform matrix
        XX (npt.NDArray): X coordinates
        YY (npt.NDArray): Y coordinates

    Returns:
        npt.NDArray: Color values
    """
    if order == 1:
        return np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(XX.shape)
    return np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX**2, YY**2], C)


def find_transform_matrix(data: npt.NDArray, order: int) -> npt.NDArray:
    """Finds the matrix C that can be used to convert a XY coordinate to a Color value

    Args:
        data (npt.NDArray): Sample points 

    Returns:
        npt.NDArray: C matrix
    """
    if order == 1:
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    else:
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(
            data[:, :2], axis=1), data[:, :2]**2]

    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    return C


def color_at_xy(Cs, x: int, y: int, order: int) -> npt.NDArray[np.uint8]:
    return np.array([TransformFunc(C, np.array([x]),
                                   np.array([y]), order)
                     for C in Cs]).flatten().astype(np.uint8)


def get_mesh_grids(datas, Cs, order: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # regular grid covering the domain of the data
    N = 16
    data = datas[0]
    X = data[:, 0]
    Y = data[:, 1]
    X_MG, Y_MG = np.meshgrid(
        np.linspace(min(X)-100, max(X)+100, N),
        np.linspace(min(Y)-100, max(Y)+100, N)
    )

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

        ax = fig.add_subplot(2, 3, i+1, projection='3d')
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

        ax = fig.add_subplot(2, 3, i+1+3)
        ax.contourf(X_MG, Y_MG, Z_MG)


def fitSurface(shapes: Shapes, order: int, useLocked: bool = True) -> Tuple[list[npt.NDArray], npt.NDArray, npt.NDArray]:
    if useLocked:
        shapeList = shapes.locked
    else:
        shapeList = shapes.all

    XYR = np.array([[s.centerX, s.centerY, s.color[2]] for s in shapeList])
    XYG = np.array([[s.centerX, s.centerY, s.color[1]] for s in shapeList])
    XYB = np.array([[s.centerX, s.centerY, s.color[0]] for s in shapeList])

    datas = [XYB, XYG, XYR]
    Cs = [find_transform_matrix(d, order) for d in datas]
    MGs = get_mesh_grids(datas, Cs, order)
    return datas, MGs, Cs


def setShapeColorEstimations(shapes: Shapes, Cs: npt.NDArray, order: int):
    shapes.close_to_estimate.clear()
    for shape in shapes.all:
        shape.colorEst = color_at_xy(Cs, shape.centerX, shape.centerY, order)
        # if shape.distToEstimation < 5:
        #     shapes.close_to_estimate.append(shape)


def get_datas(shapes: Shapes):
    XYB = [[shape.centerX, shape.centerY, shape.color[0]] for shape in shapes.all]
    XYG = [[shape.centerX, shape.centerY, shape.color[1]] for shape in shapes.all]
    XYR = [[shape.centerX, shape.centerY, shape.color[2]] for shape in shapes.all]
    return XYB, XYG, XYR
    
def get_largest_error_shape(shapes: list[Shapes]) -> Shape:
    return max(shapes, key=lambda s: s.distToEstimation)