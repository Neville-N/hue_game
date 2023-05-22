import cv2
import ownFuncs.funcs as of
from ownFuncs.shape import Shape
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as plotter
import numpy as np
import ownFuncs.solveStrategies as ss

# run settings
Npuzzle = "15"
grabSolved = False
skipSwap = False

# load image
if grabSolved:
    src = f"data/hue_solved_{Npuzzle}.png"
    Npuzzle += "s"
else:
    src = f"data/hue_scrambled_{Npuzzle}.png"
img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)

shapes = Shapes(img, Npuzzle)

print("Start swapping?")
cv2.waitKey(1)

# Pick strategy for solving puzzle
if not skipSwap:
    ss.solve3(shapes, Npuzzle)

# Wait with plotting before crashing opencv image window
print("Draw plots? opencv window will stop reacting.")
cv2.waitKey(0)

# Show plots for analyzing results
plotter.colSpacePlot(shapes.all, drawConnections=True)
plotter.surfacePlot2(shapes.all)


print("done")
