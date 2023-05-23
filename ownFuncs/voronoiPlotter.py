import cv2
import numpy as np
import ownFuncs.funcs as of

# from ownFuncs.shapes import Shapes


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def draw_voronoi(img, subdiv, shapes, draw_lines=False, draw_centroids=False):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet_arr = [f for f in facets[i]]

        ifacet = np.array(ifacet_arr).astype(np.int32)
        color = shapes.all[i].color

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        if draw_lines:
            ifacets = np.array([ifacet])
            cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        center = centers[i].astype(np.int32)
        if shapes.all[i].hardLocked or draw_centroids:
            cv2.circle(
                img,
                center,
                3,
                (0, 0, 0),
                cv2.FILLED,
                cv2.LINE_AA,
                0,
            )
    # cv2.imshow("voronoi img", img)
    # cv2.waitKey(0)
