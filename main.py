import cv2
import ownFuncs.funcs as of
from ownFuncs.shape import Shape
from ownFuncs.colorspacePlotter import colSpacePlot
import numpy as np


# load image
Npuzzle = 3
# src = f'data/hue_solved_{Npuzzle}.png'
src = f'data/hue_scrambled_{Npuzzle}.png'
img = cv2.imread(src)
assert img is not None, "file could not be read, check with os.path.exists()"

# Reduce image size to ease computations
img = of.scaleImg(img, maxHeight=1000, maxWidth=3000)
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Count colors
colors, freqs = np.unique(
    img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
sortInd = np.flip(np.argsort(freqs))

# sort by frequency high to low and leave out black
freqs = freqs[sortInd]
colors = colors[sortInd]
if sum(colors[0]) == 0:
    freqs = freqs[1:]
    colors = colors[1:]

# Only take colors that are at least 10% in size w.r.t largest color
minFreq = freqs[0]/10
mask = np.where(freqs > minFreq)
freqs = freqs[mask]
colors = colors[mask]

Shapes = {}
UnlockedShapes = []
LockedShapes: list[Shape] = []
Colors = []
ShapeMasks = {}
ShapeLoc = {}
Contours = {}
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

    maxi = np.argmax(contourAreas)
    outerContour = contours[maxi]

    # if len(contours) > 2:
    #     continue

    # if len(contours) == 0:
    #     continue

    cstr = of.arr_format(c, '3')
    Colors.append(cstr)
    ShapeMasks[cstr] = shapeMask
    ShapeLoc[cstr] = np.argwhere(shapeMask != 0)
    Contours[cstr] = contours

    locked = len(contours) > 1
    lockedShape = Shape(outerContour, mask, c, locked)
    Shapes[cstr] = lockedShape
    if locked:
        LockedShapes.append(lockedShape)
    else:
        UnlockedShapes.append(lockedShape)


collecter = np.zeros_like(img)
shapeShower = np.zeros_like(img)


def showShapes():
    arr = np.zeros_like(img)
    for s in Shapes.values():
        s.drawContour(arr)
        s.drawCentroid(arr)
    return arr


for i, c in enumerate(Colors):

    contours = Contours[c]

    imgC = img.copy()
    cv2.drawContours(imgC, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", imgC)

    col = [int(ci) for ci in colors[i]]
    cv2.drawContours(collecter, contours, contourIdx=-
                     1, color=col, thickness=cv2.FILLED)
    # cv2.imshow("collecter", collecter)

    shapeShower = showShapes()

    cv2.imshow("shapeShower", shapeShower)

    areas = of.arr_format([cv2.contourArea(c) for c in contours], ".0f")
    # print(f"num: {i:3}, cnts: {len(contours)}, areas: {areas:12}, color: {c}")

    if i == len(Colors) - 1:
        cv2.waitKey(1)
    elif len(contours) > 2:
        cv2.waitKey(0)
        print("This should not have happened")
    else:
        cv2.waitKey(1)

# Notate neighbour relationships
for color, lockedShape in Shapes.items():
    lockedShape.findNeighbours(collecter, Shapes, searchRadially=False, range=10)

# Visualize which cells are counted as neighbours
i = 0
for color, shape in Shapes.items():
    i += 1
    neighbourChecker = np.copy(collecter)
    shape.drawNeighbours(neighbourChecker, color=(0, 0, 255), thickness=6)
    shape.drawContour(neighbourChecker, color=(255, 0, 0), thickness=6)
    # cv2.imshow("c", neighbourChecker)
    # cv2.waitKey(1)
    cv2.imwrite(f"data/animation{Npuzzle}/frame_{i}.png", neighbourChecker)

print("Start swapping?")
cv2.waitKey(0)


# strat 1 solve loop with strat find suitable neighbour for locked cell that previously only had 1 unlocked neighbour
loopcount = 0
stepcount = 0
while len(UnlockedShapes) > 0 and loopcount < 3:
    loopcount += 1
    # for lockedShape in LockedShapes:
    neighbour: Shape
    for lockedShape in Shapes.values():
        swappers: list[Shape] = []
        if not lockedShape.locked:
            continue

        num_unlocked_neighbours = sum(
            [n.locked*-1 + 1 for n in lockedShape.neighbours])
        # print(f"color: {color}, has num unlocked neigbours {num_unlocked_neighbours}")
        if num_unlocked_neighbours != 1:
            continue
        for s in lockedShape.neighbours:
            if s.locked:
                continue
            neighbour = s

        minDist = np.inf
        for swap_shape in UnlockedShapes:
            if not neighbour.checkSwappable(swap_shape):
                continue
            rgb_dist = lockedShape.RGB_distance(swap_shape)
            if rgb_dist < minDist:
                minDist = rgb_dist
                closestShape = swap_shape
        neighbour.swap(closestShape)

        swappers = [neighbour, closestShape]

        neighbour.locked = True
        LockedShapes.append(neighbour)
        UnlockedShapes.remove(neighbour)

        print("swap")
        shapeShower = showShapes()
        for swapper in swappers:
            swapper.drawCentroid(shapeShower, size=5, col=(255, 0, 0))
        cv2.imwrite(f"data/solveanimation{Npuzzle}/step_{stepcount}.png", shapeShower)
        stepcount += 1
        cv2.imshow("shapeShower", shapeShower)
        cv2.waitKey(1)



cv2.waitKey(0)


colSpacePlot(Shapes, drawConnections=1)


print('done')
