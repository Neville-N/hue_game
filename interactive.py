from ppadb.client import Client as AdbClient
import cv2
import ownFuncs.funcs as of
import ownFuncs.optimizationFuncs as opt
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as csplt
import argparse
import time


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode",
    )
    parser.add_argument(
        "--debugPlots",
        action="store_true",
        help="Enable saving plots in debugging mode",
    )
    parser.add_argument(
        "--csplot",
        action="store_true",
        help="Draws a few colorspace plots during process",
    )
    parser.add_argument(
        "--notSolveByShape",
        action="store_true",
        help="If set then during the the solving the shapes are not grouped by shape",
    )
    parser.add_argument(
        "--delayTime",
        type=float,
        help="Time delay between screen presses in seconds",
        default=0.01,
    )
    parser.add_argument(
        "--gradientImg",
        action="store_true",
        help="Creates interpolated gradient version of image. Takes a while to complete",
    )
    parser.add_argument(
        "--gradientImgNoMask",
        action="store_false",
        help="Creates interpolated gradient version of image. Takes a while to complete",
    )
    parser.add_argument(
        "--fastMode",
        action="store_true",
        help="Skip slow convex hull mode to move pieces faster",
    )
    args = parser.parse_args()
    # args.debug = args.debug or args.debugPlots
    return args


if __name__ == "__main__":
    startTime = time.time()
    prevTime = startTime
    # Parse the arguments
    args = parseArguments()

    if args.debug:
        of.cleanDir("data/solveanimation")
        of.cleanDir("data/debug_imgs")

    if args.debugPlots:
        of.cleanDir("data/plots")

    # Connect to phone adb
    client = AdbClient(host="127.0.0.1", port=5037)
    devices = client.devices()
    device = devices[0]

    # create and save sreenshot
    result = device.screencap()
    src = "data/hue_scrambled_screen.png"
    with open(src, "wb") as fp:
        fp.write(result)
    fp.close()
    PUZZLE_ID = "screen"

    img = cv2.imread(src)
    assert img is not None, "file could not be read"
    scaler = 0.5
    img = of.scaleImg(img, scaler)

    shapes = Shapes(img, PUZZLE_ID, reduce_factor=scaler, debug=args.debug)

    curTime = time.time()
    duration = curTime - prevTime
    prevTime = curTime
    print(f"shapes detected in {duration:.4g}s")

    print(
        f"Solving puzzle with {len(shapes.all)} shapes of which {len(shapes.unlocked)} are unlocked"
    )

    if args.csplot:
        csplt.ionshow()
        csplt.rgb_space_plot(shapes.all)
        csplt.rgb_space_plot(shapes.all, space="HSV", title="HSV plot")
        csplt.rgb_space_plot(shapes.all, space="LAB", title="LAB plot")
        csplt.xy_rgb_space_plot(shapes.all, title="Before ordening")
        csplt.surfacePlot2(shapes.all)

    order: int = 1
    datas, MGs, Cs = opt.fitSurface(shapes, order)
    opt.setShapeColorEstimations(shapes, Cs, order)

    stepcount: int = -1
    num_free_shapes = len(shapes.unlocked)
    remaining_color_est_error = 0
    if args.csplot:
        opt.plotSurfaces(datas, MGs, saveFig=True, order=1, step=1)

    while len(shapes.unlocked) > 1 and True:
        stepcount += 1
        shape = max(shapes.unlocked, key=lambda s: s.distToEstimation)
        BGR = opt.color_at_xy(Cs, shape.tapX, shape.tapY, order)
        closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
        remaining_color_est_error += err
        shapes.swapShapes(shape, closestShape)

        if not shape.same_as(closestShape) and args.debug:
            shapes.markSwappedShapes(shape, closestShape)
            of.saveImg(shapes.img, "data/solveanimation/", f"order1_{stepcount}.png")

    order = 2
    somethingChanged = True
    limit = 0

    refit = False
    while limit < 2 * len(shapes.all) and True:
        if limit % 10 == 0 or len(shapes.unlocked) == 0 or not somethingChanged:
            shapes.reset_locks()
            opt.determine_close_to_estimate(shapes)
            datas2, MGs, Cs = opt.fitSurface(shapes, order, False)
            if args.csplot:
                opt.plotSurfaces(datas2, MGs, saveFig=True, order=2, step=stepcount)
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
            if args.debug:
                shapes.markSwappedShapes(shape, closestShape)
                of.saveImg(
                    shapes.img, "data/solveanimation/", f"order2_{stepcount}.png"
                )
    curTime = time.time()
    duration = curTime - prevTime
    prevTime = curTime
    print(f"puzzle positions found in {duration:.4g}s")

    if args.gradientImg:
        opt.cvColGradient(shapes, mask=args.gradientImgNoMask)
        curTime = time.time()
        duration = curTime - prevTime
        prevTime = curTime
        print(f"gradient made in {duration:.4g}s")

    if args.debug:
        shapes.updateImg(False)
        of.saveImg(shapes.img, "data/solveanimation/", f"end1_{stepcount}.png")
        shapes.draw_voronoi(draw_lines=True, draw_centroids=True)
        of.saveImg(shapes.voronoi_img, "data/voronoi/", f"P{PUZZLE_ID}.png")
        if args.debugPlots:
            csplt.surfacePlot2(shapes.all, saveFig=True, step=stepcount)

    if args.csplot:
        shapes.reset_locks()
        csplt.xy_rgb_space_plot(shapes.all, title="After ordening")
        csplt.show()

    shapes.reset_locks()
    shapes.reset_colors()

    stepcount = -1
    conv_hull_first = None

    print("Move pieces on device")
    while len(shapes.unlocked) > 1 and stepcount < 2 * len(shapes.all):
        if args.notSolveByShape:
            shapeGroup = shapes.unlocked
        else:
            shapeGroup = shapes.get_largest_shapes()
        shape = min(shapeGroup, key=lambda s: s.tapX)
        while len(shapeGroup) > 0:
            if args.fastMode:
                shape = shapeGroup.pop()
            else:
                shape = shapes.next_convex_hull_shape(shape, shapeGroup)
            stepcount += 1
            of.progress_print(stepcount + 1, num_free_shapes)
            swap_shape = shapes.find_shape_by_color(shape.end_color)
            if swap_shape is not None:
                if not shape.same_as(swap_shape):
                    shapes.swapShapes(shape, swap_shape, device, args.delayTime)
                else:
                    # mark as locked but dont fysically swap
                    shapes.swapShapes(shape, swap_shape)

            if shape in shapeGroup:
                shapeGroup.remove(shape)

            if args.debug:
                shapes.markSwappedShapes(shape, swap_shape)
                shapes.draw_convex_hull()
                of.saveImg(
                    shapes.img,
                    "data/solveanimation/",
                    f"step_{stepcount}.png",
                )

            if args.debugPlots:
                csplt.surfacePlot2(shapes.all, saveFig=True, step=stepcount)

    curTime = time.time()
    duration = curTime - prevTime
    prevTime = curTime
    print(f"Pieces moved in {duration:.4g}s")

    if args.debug:
        shapes.updateImg(False)
        shapes.draw_convex_hull()
        of.saveImg(shapes.img, "data/solveanimation/", f"end2_{stepcount}.png")

    print("")
    duration = time.time() - startTime
    print(f"Solved in {duration:.4g}s")
