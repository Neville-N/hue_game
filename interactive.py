from ppadb.client import Client as AdbClient
import cv2
import ownFuncs.funcs as of
import ownFuncs.optimizationFuncs as opt
from ownFuncs.shapes import Shapes

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
    print(f"swap order1, same?: {shape.same_as(closestShape)}")
    shapes.swapShapes(shape, closestShape, device)

    if not shape.same_as(closestShape):
        shapes.markSwappedShapes(shape, closestShape)
        of.saveImg(
            shapes.img,
            f"data/solveanimation/P{PUZZLE_ID}/",
            f"step_{stepcount}.png",
        )

order = 2
somethingChanged = True
limit = 0

refit = False
while limit < 100:
    if limit % 10 == 0 or len(shapes.unlocked) == 0 or not somethingChanged:
        shapes.resetLocks()
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
    BGR = opt.color_at_xy(Cs, shape.centerX, shape.centerY, order)
    closestShape, err = shapes.findShapeClosestToColor(BGR, shape)
    print(f"swap order 2, same?: {shape.same_as(closestShape)}")
    shapes.swapShapes(shape, closestShape, device)

    shapes.markSwappedShapes(shape, closestShape)

    # Check if shape has swapped with itself
    if not shape.same_as(closestShape):
        somethingChanged = True
        of.saveImg(
            shapes.img,
            f"data/solveanimation/P{PUZZLE_ID}/",
            f"step_{stepcount}.png",
        )

shapes.updateImg(False)
of.saveImg(shapes.img, f"data/solveanimation/P{PUZZLE_ID}/", f"step_{stepcount}.png")

shapes.draw_voronoi(draw_lines=True, draw_centroids=True)
of.saveImg(shapes.voronoi_img, "data/voronoi/", f"P{PUZZLE_ID}.png")
