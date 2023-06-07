from ppadb.client import Client as AdbClient
import cv2
import ownFuncs.funcs as of
import ownFuncs.optimizationFuncs as opt
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as csplt

DEBUG = 1
CSPLOT = 0

if DEBUG:
    of.cleanDir("data/solveanimation")
    of.cleanDir("data/debug_imgs")

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
img = of.scaleImg(img, 0.5)
assert img is not None, "file could not be read"

shapes = Shapes(img, PUZZLE_ID, reduce_factor=0.5)
print(
    f"solving puzzle with {len(shapes.all)} shapes of which {len(shapes.unlocked)} are unlocked"
)

if CSPLOT:
    csplt.ionshow()
    csplt.rgb_space_plot(shapes.all)
    csplt.rgb_space_plot(shapes.all, space="HSV", title="HSV plot")
    csplt.rgb_space_plot(shapes.all, space="LAB", title="LAB plot")
    csplt.xy_rgb_space_plot(shapes.all, title="Before ordening")

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

order = 2
somethingChanged = True
limit = 0

refit = False
while limit < 2 * len(shapes.all) and True:
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
            of.saveImg(shapes.img, "data/solveanimation/", f"order2_{stepcount}.png")


if DEBUG:
    shapes.updateImg(False)
    of.saveImg(shapes.img, "data/solveanimation/", f"end1_{stepcount}.png")
    shapes.draw_voronoi(draw_lines=True, draw_centroids=True)
    of.saveImg(shapes.voronoi_img, "data/voronoi/", f"P{PUZZLE_ID}.png")

if CSPLOT:
    shapes.reset_locks()
    csplt.xy_rgb_space_plot(shapes.all, title="After ordening")
    csplt.show()


# shapes.define_new_centers()
# counter = 0
# while shapes.snape_to_grid_x() and counter < 20:
#     counter += 1
#     pass
# counter = 0
# while shapes.snape_to_grid_y() and counter < 20:
#     counter += 1
#     pass
# shapes.make_symmetric_x()
# shapes.make_symmetric_y()

shapes.reset_locks()
shapes.reset_colors()
shapes.sort_unlocked()

stepcount = -1
conv_hull_first = None

shape = min(shapes.unlocked, key=lambda s: s.tapX+s.tapY)
while len(shapes.unlocked) > 1 and stepcount < 2 * len(shapes.all):
    # shape = shapes.unlocked[0]
    shape = shapes.next_convex_hull_shape(shape)
    stepcount += 1
    swap_shape = shapes.find_shape_by_color(shape.end_color)
    if swap_shape is not None:
        if not shape.same_as(swap_shape):
            shapes.swapShapes(shape, swap_shape, device, 0.01)
        else:
            # mark as locked but dont fysically swap
            shapes.swapShapes(shape, swap_shape)

    if DEBUG:
        shapes.markSwappedShapes(shape, swap_shape)
        shapes.draw_convex_hull()
        of.saveImg(
            shapes.img,
            "data/solveanimation/",
            f"step_{stepcount}.png",
        )

if DEBUG:
    shapes.updateImg(False)
    shapes.draw_convex_hull()
    of.saveImg(shapes.img, "data/solveanimation/", f"end2_{stepcount}.png")
