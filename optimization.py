"""Solves hue game puzzle with optimization strategy"""
from typing import Final

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as matplotlib_use

import ownFuncs.funcs as of
import ownFuncs.optimizationFuncs as opt
from ownFuncs.shapes import Shapes
from ownFuncs.shape import Shape

matplotlib_use('TkAgg')


def optimization_strat(PUZZLE_ID: str, show_plot: bool):
    src = f'data/hue_scrambled_{PUZZLE_ID}.png'
    # src = f'data/hue_solved_{Npuzzle}.png'
    img = cv2.imread(src)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Reduce image size to ease computations
    img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)

    shapes = Shapes(img, PUZZLE_ID)
    order: int = 1
    datas, MGs, Cs = opt.fitSurface(shapes, order)
    opt.setShapeColorEstimations(shapes, Cs, order)

    stepcount: int = -1
    num_free_shapes = len(shapes.unlocked)
    remaining_color_est_error = 0

    while len(shapes.unlocked) > 1:
        stepcount += 1
        shape = max(shapes.unlocked, key=lambda s: s.distToEstimation)
        BGR = opt.color_at_xy(Cs, shape.centerX, shape.centerY, order)
        closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
        remaining_color_est_error += err
        shapes.swapShapes(shape, closestShape)
        if not np.all(np.equal(shape.colorA, closestShape.colorA)):
            shapes.markSwappedShapes(shape, closestShape)
            of.saveImg(
                shapes.img, f"data/solveanimation/P{PUZZLE_ID}/", f"step_{stepcount}.png")
    print(
        f"1st avg remaining color error: {remaining_color_est_error/num_free_shapes:.5g}")
    XYB, XYG, XYR = opt.get_datas(shapes)
    datas2 = np.array([XYB, XYG, XYR])

    if show_plot:
        opt.plotSurfaces(datas, MGs, datas2)

    order = 2
    somethingChanged = True
    limit = 0

    avg_est_error = shapes.average_estimation_error
    # somethingChanged and
    while limit < 100 and avg_est_error > 1.6:
        if limit % 10 == 0 or len(shapes.unlocked) == 0 or not somethingChanged:
            shapes.resetLocks()
            opt.determine_close_to_estimate(shapes)
            datas2, MGs, Cs = opt.fitSurface(shapes, order, False)
            opt.setShapeColorEstimations(shapes, Cs, order)
            print(f"{'-'*20} refit {'-'*20}")

        somethingChanged = False
        limit += 1
        stepcount += 1

        shape = opt.get_largest_error_shape(shapes.unlocked)
        BGR = opt.color_at_xy(Cs, shape.centerX, shape.centerY, order)
        closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
        shapes.swapShapes(shape, closestShape)
        shapes.markSwappedShapes(shape, closestShape)

        # Check if shape has swapped with itself
        if not shape.same_as(closestShape):
            somethingChanged = True
            of.saveImg(
                shapes.img, f"data/solveanimation/P{PUZZLE_ID}/", f"step_{stepcount}.png")

        avg_est_error = shapes.average_estimation_error

        print(
            f"Average remaining color error: {avg_est_error:.5g}")
        if somethingChanged and show_plot:
            XYB, XYG, XYR = opt.get_datas(shapes)
            datas2 = np.array([XYB, XYG, XYR])
            opt.plotSurfaces(datas, MGs, datas2)

    shapes.updateImg(False)
    of.saveImg(
        shapes.img, f"data/solveanimation/P{PUZZLE_ID}/", f"step_{stepcount}.png")

    print(f"P{PUZZLE_ID} Ran order two loop for {limit} times")

    if show_plot:
        plt.show()


ids = [str(i) for i in range(16)]
ids.insert(0, '00')
ids.insert(0, '000')

for id in ids:
    optimization_strat(id, False)
    print('\n')

# optimization_strat('14', False)
