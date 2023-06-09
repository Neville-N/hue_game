import matplotlib.pyplot as plt
import numpy as np


def calcAngle(a, b, c, limit=False) -> float:
    bIsRightOfac = pointRightOfLine(a, b, c)
    angle = smallAngle(a - b, c - b)
    if bIsRightOfac:
        angle = 2 * np.pi - angle
    if limit and angle > np.pi + 0.5:
        return 0.0
    return angle


def smallAngle(v1, v2) -> float:
    """Calculate the smaller angle between vector v1 and v2

    Args:
        v1 (NDarray): xy vector
        v2 (NDarray): xy vector

    Returns:
        float : Angle between v1 and v2
    """
    return np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def pointRightOfLine(a, b, c):
    """Determines wether point c is on the right of line through a and b

    Args:
        a (NDarray): x,y coordinate of point
        b (NDarray): x,y coordinate of point
        c (NDarray): x,y coordinate of point

    Returns:
        bool: False if point is exactly on or to the left of line, else True
    """
    ca = c - a
    ba = b - a
    return ca[0] * ba[1] - ca[1] * ba[0] < 0


def point_line_dist(P0, P1, P2) -> float:
    """Calculate distance between point P0 and a line passing through P1 and P2

    Returns:
        float: distance
    """
    dist = abs(
        (P2[0] - P1[0]) * (P1[1] - P0[1]) - (P1[0] - P0[0]) * (P2[1] - P1[1])
    ) / np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)
    return dist


def pointClose2Line(A, B, C, angle_tol=0.1) -> bool:
    """Tests wether point C is close to line segment AB

    Args:
        A (NDarray): point coordinates
        B (NDarray): point coordinates
        C (NDarray): point coordinates

    Returns:
        bool: Returns True if point C is close to AB
    """
    distAB = point_dist(A, B)
    if point_dist(A, C) > distAB or point_dist(B, C) > distAB:
        return False
    return smallAngle(A - B, C - B) < angle_tol


def point_dist(P0, P1) -> float:
    """Calculate the distance betwoon two points

    Returns:
        float: distance between P0 and P1
    """
    return np.linalg.norm(P0 - P1)


rng = np.random.default_rng(1234)
N = 30
points = rng.random((N, 2))
indices = list(range(N))

convex_hull = []
concave_hull = []
leftindex = np.argmin(points, axis=0)[0]
leftpoint = points[leftindex]
indices.remove(leftindex)

pointA = leftpoint - np.array([0, 1])
pointB = leftpoint
convex_hull.append(pointB)

# find convex hull shape
while len(indices) > 0:
    pointC_i = max(indices, key=lambda pi: calcAngle(pointA, pointB, points[pi]))
    pointC = points[pointC_i]
    # winnerangle = calcAngle(pointA, pointB, pointC)
    # print(f"{winnerangle/np.pi*180:.4g}")
    indices.remove(pointC_i)
    convex_hull.append(pointC)
    pointA, pointB = pointB, pointC
convex_hull_arr = np.array(convex_hull)

# reset some vars and start looking for concave hull
indices = list(range(N))
indices.remove(leftindex)
pointA = np.array([0, 0])
pointB = leftpoint
concave_hull.append(pointB)

# find concave hull
while len(indices) > 1:
    option1 = max(
        indices, key=lambda pi: calcAngle(pointA, pointB, points[pi], limit=False)
    )
    pointC = points[option1]
    winnerangle = calcAngle(pointA, pointB, pointC)

    # print(pointA, pointB, pointC)
    option2 = -1
    angle = np.inf
    distBC = point_dist(pointB, pointC)
    for i in indices:
        angle_test = smallAngle(pointB - pointC, points[i] - pointC)
        if angle_test <= angle and pointClose2Line(pointB, pointC, points[i]):
            angle = angle_test
            option2 = i

    if option2 > 0:
        indices.remove(option2)
        pointC = points[option2]
    else:
        indices.remove(option1)

    print(f"pointA : ({pointA[0]:<#8.3g}, {pointA[1]:<#8.3g})")
    print(f"pointB : ({pointB[0]:<#8.3g}, {pointB[1]:<#8.3g})")
    print(
        f"option1: ({points[option1][0]:<#8.3g}, {points[option1][1]:<#8.3g}) angle: {180*winnerangle/np.pi:.3g}"
    )
    if option2 >= 0:
        printstr = "option2: "
        printstr += f"({points[option2][0]:<#8.3g}, {points[option2][1]:<#8.3g}) "
        printstr += f"dist: {point_line_dist(pointB, points[option1], pointC):<#8.3g} "
        printstr += f"angle : {180/np.pi*angle:.3g} deg "
        printstr += f"or {angle:.3g} rad"
        print(printstr)
    else:
        print("No valid option 2 found")
    print(f"pointC : ({pointC[0]:<#8.3g}, {pointC[1]:<#8.3g})\n")
    concave_hull.append(pointC)

    # keep point A when making concave turn so hull does not collapse
    # if np.all(np.equal(pointC, points[option2])):
    #     pointB = pointC
    # else:
    pointA, pointB = pointB, pointC

concave_hull.append(points[indices[0]])
concave_hull_arr = np.array(concave_hull)

fig, axs = plt.subplots(1, 2, figsize=(10, 8))

axs[0].plot(points[:, 0], points[:, 1], marker="o", linestyle="None")
axs[0].plot(convex_hull_arr[:, 0], convex_hull_arr[:, 1])

axs[1].plot(points[:, 0], points[:, 1], marker="o", linestyle="None")
axs[1].plot(concave_hull_arr[:, 0], concave_hull_arr[:, 1])

for i in range(2):
    axs[i].axis("equal")
    axs[i].grid()

plt.tight_layout()
plt.show()
