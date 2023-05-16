from __future__ import annotations
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
        self.area = cv2.contourArea(contour)

    @property
    def colorA(self): 
        return np.array(self.color)

    def drawContour(self, img, thickness=0, color=0):
        if thickness == 0:
            thickness = cv2.FILLED
        if color == 0:
            color = self.color
        cv2.drawContours(img, [self.contour], contourIdx=-1,
                         color=color, thickness=thickness)

    def drawCentroid(self, img, size=8, col=[0, 0, 255]):
        if self.locked and col == [0, 0, 255]:
            col = [0, 255, 0]
        cv2.circle(img, self.center, size, col, -1)

    def checkSwappable(self, otherShape):
        areaRatio = self.area/otherShape.area
        areaCheck = 0.8 < areaRatio and areaRatio < 1.2
        return areaCheck

    def swap(self, otherShape):
        if self.locked:
            print("This shape is not allowed to swap, self")
        if otherShape.locked:
            print("This shape is not allowed to swap, other")
        areaRatio = self.area/otherShape.area
        if areaRatio > 1.2 or areaRatio < 0.8:
            print(f"performing illegal swap, area ratio is {areaRatio}")
        self.color, otherShape.color = otherShape.color, self.color

    def findNeighbours(self, img, shapes: list[Shape], searchRadially=True, range=10):
        dirs = range * np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
        found_colors = [self.color, [0, 0, 0]]
        count_colors = {}

        for c in self.contour:
            c = c[0]

            if searchRadially:
                # check radial directions
                dir = c - self.center
                check_locations = [
                    c + np.int64(range / np.linalg.norm(dir) * dir)]
            else:
                # check cardinal directions
                check_locations = c + dirs

            for check_dir in check_locations:
                check_color = img[check_dir[1], check_dir[0]].tolist()
                if check_color not in found_colors:
                    if str(check_color) not in count_colors.keys():
                        count_colors[str(check_color)] = 0
                    count_colors[str(check_color)] += 1

                    if count_colors[str(check_color)] > 10:
                        # print(f"new color at: {check_dir}")
                        found_colors.append(check_color)
                        found_shape = shapes[of.arr_format(check_color, '3')]
                        self.neighbours.append(found_shape)

    def drawNeighbours(self, img, color=(0, 255, 0), thickness=3):
        for n in self.neighbours:
            n.drawContour(img, thickness=thickness, color=color)

    def RGB_distance(self, shape: Shape):
        return np.linalg.norm(self.colorA - shape.colorA)
