import cv2
import numpy as np
from ownFuncs.shape import Shape
import ownFuncs.funcs as of


class Shapes:
    def __init__(self, img: cv2.Mat, puzzleId:int):
        colors, minFreq = of.collectCollors(img)

        self.all: list[Shape] = []
        self.unlocked: list[Shape] = []
        self.locked: list[Shape] = []
        self.imgref = img
        self.img = np.zeros_like(img)

        for i, c in enumerate(colors):
            shapeMask_raw = cv2.inRange(self.imgref, c, c)

            kernel = np.ones((3, 3), np.uint8)
            if max(self.imgref.shape) > 800:
                kernel = np.ones((5, 5), np.uint8)

            dilated = cv2.dilate(shapeMask_raw, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=2)
            shapeMask = cv2.dilate(eroded, kernel, iterations=1)

            contours, hierarchy = cv2.findContours(
                shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
            self.all.append(shape)
            if locked:
                self.locked.append(shape)
            else:
                self.unlocked.append(shape)

        # recreate img with found contours
        self.updateImg(False)

        # Notate neighbour relationships
        for shape in self.all:
            shape.findNeighbours(
                self.img, self.all, searchRadially=False, range=10)

    def updateImg(self, drawCentroid=True):
        self.img = np.zeros_like(self.imgref)
        for s in self.all:
            s.drawContour(self.img)
            if drawCentroid:
                s.drawCentroid(self.img)

    def visualizeNeighbours(self):
        """Visualize which cells are counted as neighbours
        """    
        i = 0
        for shape in self.all:
            i += 1
            neighbourChecker = np.copy(self.imgref)
            shape.drawNeighbours(neighbourChecker, color=(0, 0, 255), thickness=6)
            shape.drawContour(neighbourChecker, color=(255, 0, 0), thickness=6)
            # cv2.imshow("c", neighbourChecker)
            # cv2.waitKey(1)
            cv2.imwrite(
                f"data/neighbourChecker{self.puzzleId}/frame_{i}.png", neighbourChecker)
            
    def swapShapes(self, A:Shape, B:Shape):
        A.swap(B)
        A.locked = True
        self.unlocked.remove(A)
        self.locked.append(A)
        self.updateImg()
        # swappers = [A, B]

    def resetLocks(self):
        for s in self.all:
            s.locked = s.hardLocked
            if not s.locked:
                self.locked.remove(s)
                self.unlocked.append(s)
