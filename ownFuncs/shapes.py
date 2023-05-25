import cv2
import numpy as np
from ownFuncs.shape import Shape
import ownFuncs.funcs as of
import ownFuncs.voronoiPlotter as vp
import time


class Shapes:
    def __init__(self, img: cv2.Mat, puzzleId: str = "_", search_neighbours=False):
        colors, minFreq = of.collectCollors(img)

        self.all: list[Shape] = []
        self.unlocked: list[Shape] = []
        self.locked: list[Shape] = []
        self.close_to_estimate: list[Shape] = []
        self.imgref = img
        self.img = np.zeros_like(img)
        self.voronoi_img = np.zeros_like(img)

        for i, c in enumerate(colors):
            shapeMask_raw = cv2.inRange(self.imgref, c, c)

            kernel = np.ones((3, 3), np.uint8)
            if max(self.imgref.shape) > 800:
                kernel = np.ones((5, 5), np.uint8)

            dilated = cv2.dilate(shapeMask_raw, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=2)
            shapeMask = cv2.dilate(eroded, kernel, iterations=1)

            contours, hierarchy = cv2.findContours(
                shapeMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

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
        if search_neighbours:
            for shape in self.all:
                shape.findNeighbours(self.img, self.all, searchRadially=False, range=10)

    @property
    def average_estimation_error(self) -> float:
        return sum([s.distToEstimation for s in self.all]) / len(self.all)

    @property
    def center_points(self):
        return [s.center for s in self.all]

    @property
    def centerX(self):
        return [s.centerX for s in self.all]

    @property
    def centerY(self):
        return [s.centerY for s in self.all]

    def updateImg(self, drawCentroid=True):
        self.img = np.zeros_like(self.imgref)
        for s in self.all:
            s.drawContour(self.img)
            if drawCentroid:
                s.drawCentroid(self.img)

    def visualizeNeighbours(self, saveImage=True, drawOnScreen=True):
        """Visualize which cells are counted as neighbours"""
        i = 0
        for shape in self.all:
            i += 1
            neighbourChecker = np.copy(self.imgref)
            shape.drawNeighbours(neighbourChecker, color=(0, 0, 255), thickness=6)
            shape.drawContour(neighbourChecker, color=(255, 0, 0), thickness=6)
            if drawOnScreen:
                cv2.imshow("c", neighbourChecker)
                cv2.waitKey(0)
            if saveImage:
                folder = f"data/neighbourChecker{self.puzzleId}/"
                filename = f"frame_{i}.png"
                of.saveImg(neighbourChecker, folder, filename)

    def swapShapes(self, A: Shape, B: Shape, dev=None):
        A.swap(B)
        A.locked = True
        self.unlocked.remove(A)
        self.locked.append(A)
        self.updateImg()
        if not A.same_as(B):
            if dev is not None:
                self.fysical_swap_swipe(A, B, dev)
                time.sleep(0.2)

    def fysical_swap(self, A: Shape, B: Shape, dev):
        command = f"input tap {A.centerX} {A.centerY}"
        print(command)
        dev.shell(command)

        command2 = f"input tap {B.centerX} {B.centerY}"
        print(command2)
        dev.shell(command2)

    def fysical_swap_swipe(self, A, B, dev):
        command = f"input swipe {B.centerX} {B.centerY} {A.centerX} {A.centerY} 200"
        print(command)
        dev.shell(command)

    def markSwappedShapes(self, A: Shape, B: Shape):
        self.img = cv2.arrowedLine(self.img, B.center, A.center, [0, 0, 0], 5)

    def resetLocks(self):
        for s in self.all:
            s.locked = s.hardLocked
            if s.hardLocked:
                continue
            if s in self.locked:
                self.locked.remove(s)
            if s not in self.unlocked:
                self.unlocked.append(s)

    def findShapeClosestToColor(self, BGR: np.ndarray, shape: Shape = None) -> Shape:
        mindist = np.inf
        for s in self.unlocked:
            dist = s.RGB_distance(None, BGR)
            if dist > mindist:
                continue
            if shape is not None:
                if not shape.checkSwappable(s):
                    continue
            mindist = dist
            closestShape = s
        # closestShape = min(self.unlocked, key=lambda s: s.RGB_distance(None, BGR))
        # mindist = closestShape.RGB_distance(None, BGR)
        return closestShape, mindist

    def draw_voronoi(self, f=1, draw_lines=True, draw_centroids=False):
        size = self.img.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        for s in self.all:
            cx = s.centerX
            cy = s.centerY
            # cx -= cx % f
            # cy -= cy % f
            cx = f * round(cx / f)
            cy = f * round(cy / f)

            subdiv.insert((int(cx), int(cy)))

        vp.draw_voronoi(self.voronoi_img, subdiv, self, draw_lines, draw_centroids)

    def snape_to_grid_x(self):
        unique_x, freq_x = np.unique(self.centerX, return_counts=True)
        # print(f"before X {unique_x}")
        changed = False
        rangelim = self.img.shape[0] // 100
        for i, x in enumerate(unique_x[:-1]):
            if unique_x[i + 1] - x < rangelim:
                to_x = (freq_x[i] * x + freq_x[i + 1] * unique_x[i + 1]) / (
                    freq_x[i] + freq_x[i + 1]
                )
                self.move_shapes_x(x, to_x)
                self.move_shapes_x(unique_x[i + 1], to_x)
                changed = True
        unique_x, freq_x = np.unique(self.centerX, return_counts=True)
        # print(f"after X  {unique_x}")
        return changed

    def snape_to_grid_y(self):
        unique_y, freq_y = np.unique(self.centerY, return_counts=True)
        # print(f"before Y {unique_y}")
        changed = False
        rangelim = self.img.shape[0] // 100
        for i, y in enumerate(unique_y[:-1]):
            if unique_y[i + 1] - y < rangelim:
                to_y = (freq_y[i] * y + freq_y[i + 1] * unique_y[i + 1]) / (
                    freq_y[i] + freq_y[i + 1]
                )
                self.move_shapes_y(y, to_y)
                self.move_shapes_y(unique_y[i + 1], to_y)
                changed = True
        unique_y, freq_y = np.unique(self.centerY, return_counts=True)
        # print(f"after Y  {unique_y}")
        return changed

    def move_shapes_x(self, start_x, to_x):
        for shape in self.all:
            if shape.centerX == start_x:
                shape.center[0] = to_x

    def move_shapes_y(self, start_y, to_y):
        for shape in self.all:
            if shape.centerY == start_y:
                shape.center[1] = to_y

    def make_symmetric_x(self):
        unique_x = np.unique(self.centerX)
        dx = np.diff(unique_x)
        dx += np.flip(dx)

        # switch between rounding up and down
        adder = 1
        for i, di in enumerate(dx):
            if di % 2 == 1:
                dx[i] += adder
                adder *= 1
        dx //= 2
        new_x = np.concatenate((np.array([unique_x[0]]), unique_x[0] + np.cumsum(dx)))
        for start_x, to_x in zip(unique_x, new_x):
            self.move_shapes_x(start_x, to_x)

    def make_symmetric_y(self):
        unique_y = np.unique(self.centerY)
        dy = np.diff(unique_y)
        dy += np.flip(dy)

        # switch between rounding up and down
        adder = 1
        for i, di in enumerate(dy):
            if di % 2 == 1:
                dy[i] += adder
                adder *= 1
        dy //= 2
        new_y = np.concatenate((np.array([unique_y[0]]), unique_y[0] + np.cumsum(dy)))
        for start_y, to_y in zip(unique_y, new_y):
            self.move_shapes_y(start_y, to_y)

    def sort_all(self):
        self.all = sorted(self.all, key=lambda s: s.dim1)
