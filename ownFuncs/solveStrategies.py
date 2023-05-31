from ownFuncs.shapes import Shapes
import ownFuncs.funcs as of
import cv2


def solve2(shapes: Shapes, Npuzzle: str):
    """N times find shape with most locked neighbours and pick color relative to them"""
    shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)
    loopcount = 0
    stepcount = 0
    onlyCheckLocked = True

    # while stepcount < stratSteps and loopcount < 1:
    while len(shapes.unlocked) > 1:
        stepcount += 1
        current_shape = shapes.unlocked[0]
        swap_shape = current_shape.findBestSwap(shapes.unlocked, onlyCheckLocked)
        if swap_shape:
            shapes.swapShapes(current_shape, swap_shape)
            shapes.markSwappedShapes(current_shape, swap_shape)
            print(f"{stepcount} swap")
            of.saveImg(
                shapes.img, f"data/solveanimation{Npuzzle}/", f"step_{stepcount}.png"
            )
            cv2.imshow("shapeShower", shapes.img)
            cv2.waitKey(1)
        else:
            print(f"{stepcount} no swap")
        shapes.unlocked.sort(key=lambda x: x.countLockedNeighbours, reverse=True)

        if len(shapes.unlocked) == 0:
            print("new loop")
            loopcount += 1
            onlyCheckLocked = False
            shapes.reset_locks()


def solve3(shapes: Shapes, Npuzzle: str):
    stepCount = 0
    while len(shapes.unlocked) > 1 and stepCount < 100:
        stepCount += 1
