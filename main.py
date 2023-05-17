import cv2
import ownFuncs.funcs as of
from ownFuncs.shape import Shape
from ownFuncs.colorspacePlotter import colSpacePlot, surfacePlot
import numpy as np
import threading

# run settings
Npuzzle = '1'
grabSolved = False

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

colors, minFreq = of.collectCollors(img)

Shapes: list[Shape] = []
UnlockedShapes: list[Shape] = []
LockedShapes: list[Shape] = []

for i, c in enumerate(colors):
    shapeMask_raw = cv2.inRange(img, c, c)

    kernel = np.ones((3, 3), np.uint8)
    if max(img.shape) > 800:
        kernel = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(shapeMask_raw, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    shapeMask = cv2.dilate(eroded, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # simplify contours
    # newContours = []
    # for cnt in contours:
    #     epsilon = 0.02*cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     print(f"{i:3}: {cnt.shape[0]:3} -> {approx.shape[0]}")
    # newContours.append(approx)
    # contours = newContours

    # Filter based on area again after dilation erosion process.
    contourAreas = [cv2.contourArea(c) for c in contours]
    if sum(contourAreas) < minFreq:
        continue

    if len(contours) > 2:
        print("Large amount of contours found for color")
        continue

    maxi = np.argmax(contourAreas)
    outerContour = contours[maxi]   

    locked = len(contours) > 1
    shape = Shape(contours, outerContour, shapeMask, c, locked)
    Shapes.append(shape)
    if locked:
        LockedShapes.append(shape)
    else:
        UnlockedShapes.append(shape)


def showShapes(arr, drawCentroid = False):
    for s in Shapes:
        s.drawContour(arr)
        if drawCentroid:
            s.drawCentroid(arr)
    return arr

collecter = showShapes(np.zeros_like(img))
shapeShower = showShapes(np.zeros_like(img), True)


# for i, shape in enumerate(Shapes):
#     cv2.drawContours(collecter, shape.allContours, contourIdx=-
#                      1, color=shape.color, thickness=cv2.FILLED)
#     cv2.imshow("collecter", collecter)

#     shapeShower = showShapes(shapeShower)
#     cv2.imshow("shapeShower", shapeShower)

#     if i == len(Shapes) - 1:
#         cv2.waitKey(1)
#     else:
#         cv2.waitKey(1)

# Notate neighbour relationships
for shape in Shapes:
    shape.findNeighbours(
        collecter, Shapes, searchRadially=False, range=10)

# Visualize which cells are counted as neighbours
i = 0
for shape in Shapes:
    i += 1
    neighbourChecker = np.copy(collecter)
    shape.drawNeighbours(neighbourChecker, color=(0, 0, 255), thickness=6)
    shape.drawContour(neighbourChecker, color=(255, 0, 0), thickness=6)
    # cv2.imshow("c", neighbourChecker)
    # cv2.waitKey(1)
    cv2.imwrite(f"data/neighbourChecker{Npuzzle}/frame_{i}.png", neighbourChecker)

print("Start swapping?")
cv2.waitKey(1)


# # strat 1 solve loop with strat find suitable neighbour for locked cell that previously only had 1 unlocked neighbour
# loopcount = 0
# stepcount = 0
# while len(UnlockedShapes) > 0 and loopcount < 3:
#     loopcount += 1
#     # for lockedShape in LockedShapes:
#     neighbour: Shape
#     for lockedShape in Shapes:
#         swappers: list[Shape] = []
#         if not lockedShape.locked:
#             continue

#         # print(f"color: {color}, has num unlocked neigbours {num_unlocked_neighbours}")
#         if lockedShape.countUnlockedNeighbours != 1:
#             continue
#         for s in lockedShape.neighbours:
#             if s.locked:
#                 continue
#             neighbour = s

#         minDist = np.inf
#         for swap_shape in UnlockedShapes:
#             if not neighbour.checkSwappable(swap_shape):
#                 continue
#             rgb_dist = lockedShape.RGB_distance(swap_shape)
#             if rgb_dist < minDist:
#                 minDist = rgb_dist
#                 closestShape = swap_shape
#         neighbour.swap(closestShape)

#         swappers = [neighbour, closestShape]

#         neighbour.locked = True
#         LockedShapes.append(neighbour)
#         UnlockedShapes.remove(neighbour)

#         print("swap")
#         shapeShower = showShapes()
#         for swapper in swappers:
#             swapper.drawCentroid(shapeShower, size=5, col=(255, 0, 0))
#         cv2.imwrite(
#             f"data/solveanimation{Npuzzle}/step_{stepcount}.png", shapeShower)
#         stepcount += 1
#         cv2.imshow("shapeShower", shapeShower)
#         cv2.waitKey(1)

# strat 2: N times find shape with most locked neighbours and pick color relative to them
UnlockedShapes.sort(key=lambda x: x.countLockedNeighbours, reverse=True)
loopcount = 0
stepcount = 0
onlyCheckLocked = True
stratSteps = 4000

while stepcount < stratSteps and loopcount < 1:
    stepcount += 1
    ul = UnlockedShapes[0]
    swap_shape = ul.findBestSwap(UnlockedShapes, onlyCheckLocked)
    if swap_shape:
        ul.swap(swap_shape)
        ul.locked = True
        UnlockedShapes.remove(ul)
        LockedShapes.append(ul)
        swappers = [ul, swap_shape]

        print(f"{stepcount} swap")
        shapeShower = showShapes(np.zeros_like(img), True)
        for swapper in swappers:
            swapper.drawCentroid(shapeShower, size=5, col=(255, 0, 0))
        cv2.imwrite(
            f"data/solveanimation{Npuzzle}/step_{stepcount}.png", shapeShower)
        stepcount += 1
        cv2.imshow("shapeShower", shapeShower)
        cv2.waitKey(1)
    else:
        print(f"{stepcount} no swap")
    if stepcount % 1 == 0:
        UnlockedShapes.sort(
            key=lambda x: x.countLockedNeighbours, reverse=True)

    if len(UnlockedShapes) == 0:
        print("new loop")
        loopcount += 1
        onlyCheckLocked = False
        for s in Shapes:
            s.locked = s.hardLocked
            if not s.locked:
                LockedShapes.remove(s)
                UnlockedShapes.append(s)


cv2.waitKey(0)

# t2 = threading.Thread(target=colSpacePlot, args=(Shapes, True))
# t2.start()
# t2.join()

colSpacePlot(Shapes, drawConnections=1)
surfacePlot(Shapes)


print('done')
