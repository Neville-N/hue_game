import cv2
import ownFuncs.funcs as of
from ownFuncs.shape import Shape
from ownFuncs.shapes import Shapes
import ownFuncs.colorspacePlotter as plotter
import numpy as np

# run settings
Npuzzle = '1'
grabSolved = False
skipSwap = True

# load image
if grabSolved:
    src = f'data/hue_solved_{Npuzzle}.png'
else:
    src = f'data/hue_scrambled_{Npuzzle}.png'
img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


shapes = Shapes(img, Npuzzle)

print("Start swapping?")
cv2.waitKey(1)


# strat 2: N times find shape with most locked neighbours and pick color relative to them
shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)
loopcount = 0
stepcount = 0
onlyCheckLocked = True
stratSteps = 4000


while stepcount < stratSteps and loopcount < 1:
    stepcount += 1
    ul = shapes.unlocked[0]
    swap_shape = ul.findBestSwap(shapes.unlocked, onlyCheckLocked)
    if swap_shape:
        shapes.swapShapes(ul, swap_shape)
        
        swappers = [ul, swap_shape]
        print(f"{stepcount} swap")
        
        for swapper in swappers:
            swapper.drawCentroid(shapes.img, size=5, col=[255, 0, 0])
        of.saveImg(
            shapes.img, f"data/solveanimation{Npuzzle}/", f"step_{stepcount}.png")
        cv2.imshow("shapeShower", shapes.img)
        cv2.waitKey(1)
    else:
        print(f"{stepcount} no swap")
    shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)

    if len(shapes.unlocked) == 0:
        print("new loop")
        loopcount += 1
        onlyCheckLocked = False
        shapes.resetLocks()
        


cv2.waitKey(0)

# t2 = threading.Thread(target=colSpacePlot, args=(shapes.all, True))
# t2.start()
# t2.join()

plotter.colSpacePlot(shapes.all, drawConnections=True)
plotter.surfacePlot2(shapes.all)


print('done')
