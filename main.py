import cv2
import ownFuncs.funcs as of
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as plotter
import ownFuncs.solveStrategies as ss
from ppadb.client import Client as AdbClient

# Connect to phone adb
client = AdbClient(host="127.0.0.1", port=5037)
devices = client.devices()
device = devices[0]

# run settings
PUZZLE_ID = "15"
take_screencap = True
grabSolved = False
skipSwap = False

# load image
if take_screencap:
    result = device.screencap()
    src = "data/hue_scrambled_screen.png"
    with open(src, "wb") as fp:
        fp.write(result)
    fp.close()
    PUZZLE_ID = "screen"
elif grabSolved:
    src = f"data/hue_solved_{PUZZLE_ID}.png"
    PUZZLE_ID += "s"
else:
    src = f"data/hue_scrambled_{PUZZLE_ID}.png"

img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)

shapes = Shapes(img, PUZZLE_ID)

print("Start swapping?")
cv2.waitKey(1)

# Pick strategy for solving puzzle
if not skipSwap:
    ss.solve3(shapes, PUZZLE_ID)

# Wait with plotting before crashing opencv image window
print("Draw plots? opencv window will stop reacting.")
cv2.waitKey(0)

# Show plots for analyzing results
plotter.rgb_space_plot(shapes.all)
plotter.surfacePlot2(shapes.all)


print("done")
