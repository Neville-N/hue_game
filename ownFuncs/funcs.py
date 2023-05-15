import cv2
import numpy as np


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
