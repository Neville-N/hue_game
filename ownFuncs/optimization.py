import scipy
import numpy as np
from ownFuncs.shape import Shape
from scipy.spatial.transform import Rotation

def optimize(shapes: list[Shape]):
    X = np.array([s.centerX for s in shapes])
    Y = np.array([s.centerY for s in shapes])
    R = np.array([s.color[2] for s in shapes])
    G = np.array([s.color[1] for s in shapes])
    B = np.array([s.color[0] for s in shapes])

    XYR = np.array([[s.centerX, s.centerY, s.color[2]] for s in shapes])



    def func(angles):
        rot = Rotation.from_euler('xy', angles, degrees=True)
        height = 0
        for v in XYR:
            V = rot[2]*v
            height += V
        return height
