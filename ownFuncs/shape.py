from __future__ import annotations
import numpy as np
import cv2
import ownFuncs.funcs as of
from typing import List, Final


class Shape:
    def __init__(self, allContours: np.ndarray, contour: np.ndarray, mask: cv2.Mat, color, locked: bool):
        self.allContours = allContours
        self.contour = contour
        self.color = [int(c) for c in color]

        self.locked: bool = locked
        self.hardLocked: Final = locked
        self.neighbours: list[Shape] = []

        self.mask = mask
        self.cstring = of.arr_format(color)

        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        self.center = [cx, cy]
        self.area = cv2.contourArea(contour)

    @property
    def centerX(self):
        return self.center[0]

    @property
    def centerY(self):
        return self.center[1]

    @property
    def colorA(self):
        return np.array(self.color)

    @property
    def colorHsv(self):
        return cv2.cvtColor(np.uint8([[self.color]]), cv2.COLOR_BGR2HSV)[0, 0, :]

    @property
    def countLockedNeighbours(self):
        return sum([n.locked*1 for n in self.neighbours])

    @property
    def countUnlockedNeighbours(self):
        return len(self.neighbours) - self.countLockedNeighbours

    def drawContour(self, img: cv2.Mat, thickness: int = 0, color=0):
        if thickness == 0:
            thickness = cv2.FILLED
        if color == 0:
            color = self.color
        cv2.drawContours(img, self.allContours, contourIdx=-1,
                         color=color, thickness=thickness)

    def drawCentroid(self, img, size=8, col=[0, 0, 255]):
        if self.locked and col == [0, 0, 255]:
            col = [0, 255, 0]
        cv2.circle(img, self.center, size, col, -1)

    def checkSwappable(self, otherShape: Shape) -> bool:
        """Determines if a swap is allowable between self and otherShape

        Args:
            otherShape (Shape): Potential candidate for color swap

        Returns:
            bool: If a swap is allowable between self and otherShape
        """
        areaRatio = self.area/otherShape.area
        areaCheck = 0.8 < areaRatio and areaRatio < 1.2
        return areaCheck

    def swap(self, otherShape: Shape):
        """Swaps color between this and other shape.

        Args:
            otherShape (Shape): Another shape instance to swap colors with.
        """
        if self.locked:
            print("This shape is not allowed to swap, self")
        if otherShape.locked:
            print("This shape is not allowed to swap, other")
        areaRatio = self.area/otherShape.area
        if areaRatio > 1.2 or areaRatio < 0.8:
            print(f"performing illegal swap, area ratio is {areaRatio}")

        self.color, otherShape.color = otherShape.color, self.color

    def findNeighbours(self, img: cv2.Mat, shapes: List[Shape], searchRadially: bool = True, range: int = 10):
        """Determines which shapes are close enough to be listed as "Neighbours"

        Args:
            img (cv2.Mat): Image of the shapes 
            shapes (list[Shape]): List of all shapes which could be a neighbour.  
            searchRadially (bool, optional): Determines search strategy. If false cardinal directions will be used from the contour. Defaults to True.
            range (int, optional): How far to look away from contour to hop over the black border. Defaults to 10.
        """

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
                        found_colors.append(check_color)
                        # found_shape = shapes[of.arr_format(check_color, '3')]
                        found_shape = next(
                            (s for s in shapes if s.color == check_color), None)
                        self.neighbours.append(found_shape)

    def drawNeighbours(self, img: cv2.Mat, color=[0, 255, 0], thickness=3):
        """Mark the contours of neighbours of this cell. Can be used for easy debugging.

        Args:
            img (cv2.Mat): image to draw the contour on
            color (tuple, optional): Drawn contour color. Defaults to (0, 255, 0).
            thickness (int, optional): Thickness of drawn contour. Defaults to 3.
        """
        for n in self.neighbours:
            n.drawContour(img, thickness=thickness, color=color)

    def RGB_distance(self, shape: Shape) -> float:
        """Calculates euclidiean rgb distance between self and shape.

        Args:
            shape (Shape): Shape to be calculated distance to.

        Returns:
            float: Euclidean rgb distance
        """
        return np.linalg.norm(self.colorA - shape.colorA)

    def RGB2NeighboursDistance(self, shape: Shape, onlyCheckLocked: bool = True):
        dist = 0
        counted = 0
        for n in self.neighbours:
            if onlyCheckLocked and not n.locked:
                continue
            counted += 1
            dist += shape.RGB_distance(n)
        if counted == 0:
            return np.inf
        return dist/counted

    def findBestSwap(self, shapes: list[Shape], distOnlyCheckLocked: bool = True) -> Shape:
        """Finds other shape with color that would be best suited for this shape. 
        Determines fit by calculating average rgb distance to (locked) neighbouring cells

        Args:
            shapes (list[Shape]): Collection of potential shapes to swap colors with.
            distOnlyCheckLocked (bool, optional): Determines if the distance of non locked neighbours should be ignored. Defaults to True.

        Returns:
            Shape: Shape with color closest to neighbours
        """
        minDist = np.inf
        closestShape = None
        for swap_candidate in shapes:
            if not self.checkSwappable(swap_candidate):
                continue
            rgb_dist = self.RGB2NeighboursDistance(
                swap_candidate, distOnlyCheckLocked)
            if rgb_dist < minDist:
                minDist = rgb_dist
                closestShape = swap_candidate
        if closestShape is None:
            print("No closestshape found")
        return closestShape
