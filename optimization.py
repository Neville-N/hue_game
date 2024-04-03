"""Solves hue game puzzle with optimization strategy"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as matplotlib_use

import ownFuncs.funcs as of
import ownFuncs.optimizationFuncs as opt
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as csplt

matplotlib_use("TkAgg")

DEBUG = True

if DEBUG:
    of.cleanDir("data/solveanimation")
    of.cleanDir("data/debug_imgs")

logstr: str = ""
log_file = open("log.txt", "w")
log_file.write(logstr)
log_file.close()
log_file = open("log.txt", "a")


def log(text: str):
    global log_file
    log_file.write(f"\n{text}")


def optimization_strat(
    PUZZLE_ID: str,
    show_plot: bool,
    SAVE_IMAGES: bool = True,
    scaler: float = 0.5,
):
    src = f"data/hue_scrambled_{PUZZLE_ID}.png"
    # src = f'data/hue_solved_{Npuzzle}.png'
    img = cv2.imread(src)
    assert img is not None, "file could not be read"

    # # Reduce image size to ease computations
    img = of.scaleImg(img, scaler)

    shapes = Shapes(img, PUZZLE_ID, debug=DEBUG)

    order: int = 1
    datas, MGs, Cs = opt.fitSurface(shapes, order)
    opt.setShapeColorEstimations(shapes, Cs, order)

    stepcount: int = -1
    num_free_shapes = len(shapes.unlocked)
    remaining_color_est_error = 0

    while len(shapes.unlocked) > 1 and True:
        stepcount += 1
        shape = max(shapes.unlocked, key=lambda s: s.distToEstimation)
        BGR = opt.color_at_xy(Cs, shape.tapX, shape.tapY, order)
        closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
        remaining_color_est_error += err
        shapes.swapShapes(shape, closestShape)

        if not shape.same_as(closestShape) and DEBUG:
            shapes.markSwappedShapes(shape, closestShape)
            of.saveImg(shapes.img, "data/solveanimation/", f"order1_{stepcount}.png")

        log(f"1st avg remaining color error: {shapes.average_estimation_error:.5g}")

    order = 2
    somethingChanged = True
    limit = 0

    refit = False
    while limit < 4 * len(shapes.all) and True:
        if limit % 10 == 0 or len(shapes.unlocked) == 0 or not somethingChanged:
            shapes.reset_locks()
            opt.determine_close_to_estimate(shapes)
            datas2, MGs, Cs = opt.fitSurface(shapes, order, False)
            opt.setShapeColorEstimations(shapes, Cs, order)
            if refit:
                break
            refit = True
        else:
            refit = False

        somethingChanged = False
        limit += 1
        stepcount += 1

        shape = opt.get_largest_error_shape(shapes.unlocked)
        BGR = opt.color_at_xy(Cs, shape.tapX, shape.tapY, order)
        closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
        shapes.swapShapes(shape, closestShape)

        # Check if shape has swapped with itself
        if not shape.same_as(closestShape):
            somethingChanged = True
            if DEBUG:
                shapes.markSwappedShapes(shape, closestShape)
                of.saveImg(
                    shapes.img, "data/solveanimation/", f"order2_{stepcount}.png"
                )
        log(f"Average remaining color error: {shapes.average_estimation_error:.5g}")

    # while len(shapes.unlocked) > 1:
    #     stepcount += 1
    #     shape = max(shapes.unlocked, key=lambda s: s.distToEstimation)
    #     BGR = opt.color_at_xy(Cs, shape.centerX, shape.centerY, order)
    #     closestShape, _ = shapes.findShapeClosestToColor(BGR, shape)
    #     shapes.swapShapes(shape, closestShape)
    #     # if not np.all(np.equal(shape.colorA, closestShape.colorA)):
    #     if not shape.same_as(closestShape) and SAVE_IMAGES:
    #         shapes.markSwappedShapes(shape, closestShape)
    #         of.saveImg(
    #             shapes.img,
    #             "data/solveanimation/",
    #             f"order1_{stepcount}.png",
    #         )
    #     log(f"1st avg remaining color error: {shapes.average_estimation_error:.5g}")
    # XYB, XYG, XYR = opt.get_datas(shapes)
    # datas2 = np.array([XYB, XYG, XYR])

    # if show_plot:
    #     opt.plotSurfaces(datas, MGs, datas2)

    # order = 2
    # somethingChanged = True
    # limit = 0

    # refit = False
    # while limit < 2 * len(shapes.all):
    #     if limit % 10 == 0 or len(shapes.unlocked) == 0 or not somethingChanged:
    #         shapes.reset_locks()
    #         opt.determine_close_to_estimate(shapes)
    #         datas2, MGs, Cs = opt.fitSurface(shapes, order, False)
    #         opt.setShapeColorEstimations(shapes, Cs, order)
    #         if refit:
    #             log(f"refit break, avg err -> {shapes.average_estimation_error}")
    #             break
    #         log(f"{'-'*20} refit {'-'*20}")
    #         refit = True
    #     else:
    #         refit = False

    #     somethingChanged = False
    #     limit += 1
    #     stepcount += 1

    #     shape = opt.get_largest_error_shape(shapes.unlocked)
    #     BGR = opt.color_at_xy(Cs, shape.centerX, shape.centerY, order)
    #     closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
    #     shapes.swapShapes(shape, closestShape)
    #     shapes.markSwappedShapes(shape, closestShape)

    #     # Check if shape has swapped with itself
    #     if not shape.same_as(closestShape) and SAVE_IMAGES:
    #         somethingChanged = True
    #         of.saveImg(
    #             shapes.img,
    #             "data/solveanimation/",
    #             f"order2_{stepcount}.png",
    #         )

    #     log(f"Average remaining color error: {shapes.average_estimation_error:.5g}")
    #     if somethingChanged and show_plot:
    #         XYB, XYG, XYR = opt.get_datas(shapes)
    #         datas2 = np.array([XYB, XYG, XYR])
    #         opt.plotSurfaces(datas, MGs, datas2)

    if DEBUG:
        XYB, XYG, XYR = opt.get_datas(shapes)
        datas2 = np.array([XYB, XYG, XYR])
        opt.plotSurfaces(datas, MGs, datas2)
        csplt.rgb_space_plot(shapes.all, False)
        plt.show()

    shapes.updateImg(False)
    of.saveImg(shapes.img, "data/solveanimation/", f"step_{stepcount}.png")

    log(f"P{PUZZLE_ID} Ran order two loop for {limit} times")

    if show_plot:
        plt.show()

    shapes.draw_voronoi(draw_lines=True, draw_centroids=True)
    of.saveImg(shapes.voronoi_img, "data/voronoi/", f"P{PUZZLE_ID}.png")


# for id in [str(i) for i in range(34)]:
#     print(id)
#     optimization_strat(id, False, True)
#     log("\n")


optimization_strat("goes_wrong", False)

# plt.show()

log_file.close()
