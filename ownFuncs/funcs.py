import cv2
import numpy as np
import os

def scaleImg(img, factor: float = 1., maxWidth: int = 1, maxHeight: int = 1, interpolation=cv2.INTER_LINEAR):
    width = img.shape[1]
    height = img.shape[0]

    if factor != 1:
        pass
    elif maxWidth != 1 and maxHeight != 1:
        widthFactor = maxWidth/width
        heightFactor = maxHeight/height
        factor = min(widthFactor, heightFactor)
    elif maxWidth != 1:
        factor = maxWidth/width
    elif maxHeight != 1:
        factor = maxHeight/height

    width = int(width*factor)
    height = int(height*factor)
    return cv2.resize(img, (width, height), interpolation=interpolation)

def collectCollors(img: cv2.Mat):
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
    return colors, minFreq

def saveImg(img: cv2.Mat, dir: str, filename: str):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    cv2.imwrite(dir+filename, img)

def arr_format(arr, format=" "):
    formatted_arr = [f"{v:{format}}" for v in arr]

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


