import ownFuncs.funcs as of
from ownFuncs.shape import Shape
# from ownFuncs.colorspacePlotter import colSpacePlot
import numpy as np
from matplotlib import use as matplotlib_use
matplotlib_use('TkAgg')
import matplotlib.pyplot as plt
import cv2


# load image
Npuzzle = 5
src = f'data/hue_solved_{Npuzzle}.png'
# src = f'data/hue_scrambled_{Npuzzle}.png'
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
    Shapes[cstr] = (Shape(outerContour, mask, c, locked))


collecter = np.zeros_like(img)
shapeShower = np.zeros_like(img)

for i, c in enumerate(Colors):

    contours = Contours[c]

    imgC = img.copy()
    cv2.drawContours(imgC, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", imgC)

    col = [int(ci) for ci in colors[i]]
    cv2.drawContours(collecter, contours, contourIdx=-
                     1, color=col, thickness=cv2.FILLED)
    cv2.imshow("collecter", collecter)

    shape = Shapes[c]
    shape.drawContour(shapeShower)
    shape.drawCentroid(shapeShower)

    cv2.imshow("shapeShower", shapeShower)

    areas = of.arr_format([cv2.contourArea(c) for c in contours], ".0f")
    print(f"num: {i:3}, cnts: {len(contours)}, areas: {areas:12}, color: {c}")

    if i == len(Colors) - 1:
        cv2.waitKey(1)
    elif len(contours) > 2:
        cv2.waitKey(0)
        print("This should not have happened")
    else:
        cv2.waitKey(1)

i = 0
for color, shape in Shapes.items():
    i += 1
    neighbourChecker = np.copy(collecter)
    shape.findNeighbours(neighbourChecker, Shapes)
    # shape.drawNeighbours(neighbourChecker, color=(0, 0, 255), thickness=6)
    # shape.drawContour(neighbourChecker, color=(255, 0, 0), thickness=6)
    ## cv2.imshow("c", neighbourChecker)
    # #cv2.waitKey(2)
    # cv2.imwrite(f"data/animation{Npuzzle}b/frame_{i}.png", neighbourChecker)


def colSpacePlot(Shapes):
    ax = plt.figure().add_subplot(projection='3d')
    for color, shape in Shapes.items():
        for neighbour in shape.neighbours:
            ax.plot([shape.color[0], neighbour.color[0]], [shape.color[1], neighbour.color[1]], [
                shape.color[2], neighbour.color[2]], color='k')
        ax.plot(shape.color[0], shape.color[1], shape.color[2],
                marker='o', color=np.flip(np.array(shape.color)/255.0), alpha=1, ms=20)
    plt.show()


colSpacePlot(Shapes)


print('done')
