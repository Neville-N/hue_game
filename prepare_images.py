import cv2
import numpy as np
from ownFuncs.shape import Shape
import ownFuncs.funcs as of
import glob

jpgs = glob.glob('data/*.jpg')

for src in jpgs:
    img = cv2.imread(src)
    assert img is not None, "file could not be read, check with os.path.exists()"

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

    Shapes: dict[Shape] = {}
    UnlockedShapes: list[Shape] = []
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

    for i, c in enumerate(Colors):

        contours = Contours[c]

        imgC = img.copy()
        cv2.drawContours(imgC, contours, -1, (0, 255, 0), 2)
        cv2.imshow("contours", imgC)

        col = [int(ci) for ci in colors[i]]
        cv2.drawContours(collecter, contours, contourIdx=-
                        1, color=col, thickness=cv2.FILLED)


    bb = cv2.cvtColor(collecter, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(bb)
    margin = 20
    print(x, y, w, h)
    img = img[y-margin:y+h+margin, :, :]
    newsrc = src.replace('.jpg', '.png')
    cv2.imwrite(newsrc, img)