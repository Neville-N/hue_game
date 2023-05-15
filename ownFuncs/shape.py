import numpy as np
import cv2
import ownFuncs.funcs as of


class Shape:
    def __init__(self, contour, mask, color, locked):
        self.contour = contour
        self.color = [int(c) for c in color]
        self.locked = locked
        self.neighbours = []

        self.mask = mask
        self.cstring = of.arr_format(color)

        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        self.center = [cx, cy]

    def drawContour(self, img, thickness=0, color=0):
        if thickness == 0:
            thickness = cv2.FILLED
        if color == 0:
            color = self.color
        cv2.drawContours(img, [self.contour], contourIdx=-1,
                         color=color, thickness=thickness)

    def drawCentroid(self, img, size=5):
        col = (0, 0, 255)
        if self.locked:
            col = (0, 255, 0)
        cv2.circle(img, self.center, size, col, -1)

    def swap(self, otherShape):
        self.color, otherShape.color = otherShape.color, self.color

    def findNeighbours(self, img, shapes):
        range = 10
        dirs = range * np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
        found_colors = [self.color, [0, 0, 0]]
        found_colors_arr = np.array(found_colors)

        for c in self.contour:
            c = c[0]
            check_locations = c + dirs
            for check_dir in check_locations:
                check_color = img[check_dir[1], check_dir[0]].tolist()
                if check_color not in found_colors:
                    # if any(np.equal(found_colors_arr, check_color).all(1)):
                    print(f"new color at: {check_dir}")
                    found_colors.append(check_color)
                    found_shape = shapes[of.arr_format(check_color, '3')]
                    self.neighbours.append(found_shape)

    def drawNeighbours(self, img, color=(0, 255, 0)):
        for n in self.neighbours:
            n.drawContour(img, thickness=3, color=color)
