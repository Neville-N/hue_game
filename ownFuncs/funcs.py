import cv2
import numpy as np
import os
from shutil import rmtree


def scaleImg(
    img,
    factor: float = 1.0,
    maxWidth: int = 1,
    maxHeight: int = 1,
    interpolation=cv2.INTER_LINEAR,
):
    width = img.shape[1]
    height = img.shape[0]

    if factor != 1:
        pass
    elif maxWidth != 1 and maxHeight != 1:
        widthFactor = maxWidth / width
        heightFactor = maxHeight / height
        factor = min(widthFactor, heightFactor)
    elif maxWidth != 1:
        factor = maxWidth / width
    elif maxHeight != 1:
        factor = maxHeight / height

    width = int(width * factor)
    height = int(height * factor)
    return cv2.resize(img, (width, height), interpolation=interpolation)


def collectCollors(img: cv2.Mat):
    # Count colors
    colors, freqs = np.unique(
        img.reshape(-1, img.shape[-1]), axis=0, return_counts=True
    )
    sortInd = np.flip(np.argsort(freqs))

    # sort by frequency high to low and leave out background
    # it is assumed that the most common color is the background
    freqs = freqs[sortInd]
    colors = colors[sortInd]
    # if sum(colors[0]) == 0:
    freqs = freqs[1:]
    colors = colors[1:]

    # Only take colors that are at least 10% in size w.r.t largest color
    minFreq = freqs[0] / 10
    mask = np.where(freqs > minFreq)
    freqs = freqs[mask]
    colors = colors[mask]
    return colors, minFreq


def saveImg(img: cv2.Mat, dir: str, filename: str):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    cv2.imwrite(dir + filename, img)


def cleanDir(dir: str):
    if os.path.isdir(dir):
        rmtree(dir)
    else:
        os.makedirs(dir)


def arr_format(arr, format=" "):
    ret = "["
    for i, v in enumerate(arr):
        ret += f"{v:{format}}"
        if i < len(arr) - 1:
            ret += ", "
    ret += "]"
    return ret


def arr2_format(arr, format=" "):
    ret = "["
    for i, A in enumerate(arr):
        if i > 0:
            ret += " "
        ret += arr_format(A, format)
        if i < len(arr) - 1:
            ret += ",\n"
    ret += "]"
    return ret


def point_line_dist(P0, P1, P2) -> float:
    """Calculate distance between point P0 and a line passing through P1 and P2

    Returns:
        float: distance
    """
    dist = abs(
        (P2[0] - P1[0]) * (P1[1] - P0[1]) - (P1[0] - P0[0]) * (P2[1] - P1[1])
    ) / np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)
    return dist


def orientation(P, Q, R) -> int:
    """To find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    """
    val = (Q[1] - P[1]) * (R[0] - Q[0]) - (Q[0] - P[0]) * (R[1] - Q[1])
    if val == 0:
        return 0
    if val > 0:
        return 1
    else:
        return 2
