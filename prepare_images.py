import cv2
from ownFuncs.shapes import Shapes
import glob
import os

jpgs = glob.glob("data/backup/*.jpg")
overwrite = False

for src in jpgs:
    newsrc = src.replace(".jpg", ".png").replace("backup/", "")
    if os.path.isfile(newsrc) and not overwrite:
        continue

    img = cv2.imread(src)
    assert img is not None, "file could not be read, check with os.path.exists()"
    shapes = Shapes(img)

    bb = cv2.cvtColor(shapes.img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(bb)
    print(x, y, w, h)
    margin = 20
    img = img[y - margin : y + h + margin, :, :]
    cv2.imwrite(newsrc, img)
