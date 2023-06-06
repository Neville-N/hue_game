from __future__ import annotations
import numpy as np
import cv2
import ownFuncs.funcs as of
from typing import Final


class Shape:
    def __init__(
        self,
        allContours: np.ndarray,
        contour: np.ndarray,
        mask: cv2.Mat,
        color,
        locked: bool,
        reduce_factor: int = 1,
    ):
        self.allContours = allContours
        self.contour = contour
        self.color = [int(c) for c in color]
        self.start_color = self.color.copy()
        self.end_color = [0, 0, 0]

        self.locked: bool = locked
        self.hardLocked: Final = locked

        self.mask = mask
        self.cstring = of.arr_format(color)

        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        self.center = np.array([cx, cy]).astype(np.int32)
        self.center2 = np.zeros(2)
        self.area = cv2.contourArea(contour)
        self.colorEst = np.zeros(3)

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
    def end_hsv(self):
        return cv2.cvtColor(np.uint8([[self.end_color]]), cv2.COLOR_BGR2HSV_FULL)[
            0, 0, :
        ]

    @property
    def hsv_1d(self):
        return np.dot(self.colorHsv, (np.array([256**1, 256**2, 256**0])))

    @property
    def end_hsv_1d(self):
        return self.end_hsv @ np.array([256**2, 256**1, 256**0])
        # return np.dot(
        #     np.array([self.end_hsv[0], round(self.end_hsv[1] / 20), self.end_hsv[2]]),
        #     np.array([256**1, 256**1.5, 256**0]),
        # )

    @property
    def distToEstimation(self) -> float:
        return np.linalg.norm(self.colorA - self.colorEst)

    @property
    def dim1(self) -> int:
        return self.centerY + self.centerX

    @property
    def rgb_1d(self) -> int:
        # return self.color[0] * 256**2 + self.color[1] * 256**1 + self.color[2]
        # return np.sum(self.color)
        return np.sum(np.square(self.color))

    @property
    def dist_to_center(self) -> float:
        angle = np.angle(self.centerX + self.centerY*1j) / np.pi + 1.0
        return round(np.sqrt(np.sum(np.square(self.center2)))) + angle

    def drawContour(self, img: cv2.Mat, thickness: int = 0, color=0):
        if thickness == 0:
            thickness = cv2.FILLED
        if color == 0:
            color = self.color
        cv2.drawContours(
            img, self.allContours, contourIdx=-1, color=color, thickness=thickness
        )

    def drawCentroid(self, img, size=8, col=[0, 0, 255]):
        if self.locked and col == [0, 0, 255]:
            col = [0, 255, 0]
        cv2.circle(img, self.center, size, col, -1, cv2.LINE_AA)

    def checkSwappable(self, otherShape: Shape, verbose: bool = False) -> bool:
        """Determines if a swap is allowable between self and otherShape

        Args:
            otherShape (Shape): Potential candidate for color swap

        Returns:
            bool: If a swap is allowable between self and otherShape
        """
        areaRatio = self.area / otherShape.area
        areaCheck = 0.9 < areaRatio and areaRatio < 1.1
        if verbose and not areaCheck:
            print(f"performing illegal swap, area ratio is {areaRatio}")
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
        self.checkSwappable(otherShape)

        self.color, otherShape.color = otherShape.color, self.color

    def RGB_distance(self, shape: Shape, colorA: np.ndarray = None) -> float:
        """Calculates euclidiean rgb distance between self and shape.

        Args:
            shape (Shape): Shape to be calculated distance to.

        Returns:
            float: Euclidean rgb distance
        """
        if colorA is not None:
            return np.linalg.norm(self.colorA - colorA)
        return np.linalg.norm(self.colorA - shape.colorA)

    def same_as(self, other: Shape) -> bool:
        return np.all(np.equal(self.colorA, other.colorA))
