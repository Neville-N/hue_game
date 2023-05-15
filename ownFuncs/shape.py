import numpy as np
import cv2
import ownFuncs.funcs as of


class Shape:
    def __init__(self, contour, mask, color, locked):
        self.contour = contour
        self.color = [int(c) for c in color]
        self.locked = locked
        
        self.mask = mask
        self.cstring = of.arr_format(color)

        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        self.center = [cx, cy]

    def drawContour(self, img):
        cv2.drawContours(img, [self.contour], contourIdx=-1,
                         color=self.color, thickness=cv2.FILLED)

    def drawCentroid(self, img, size=5):
        col = (0, 0, 255)
        if self.locked:
            col = (0, 255, 0)
        cv2.circle(img, self.center, size, col, -1)

    def swap(self, otherShape):
        self.contour, otherShape.contour = otherShape.contour, self.contour
        self.center, otherShape.center = otherShape.center, self.center
