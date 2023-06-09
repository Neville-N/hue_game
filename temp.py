import matplotlib.pyplot as plt
import numpy as np


def calcAngle(a, b, c, limit=True):
    ab = a - b
    cb = c - b
    ca = c - a
    ba = b - a
    bIsRightOfac = ca[0] * ba[1] - ca[1] * ba[0] < 0
    angle = np.arccos(ab @ cb / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    if bIsRightOfac:
        angle = 2 * np.pi - angle
    if limit and angle > 1.1*np.pi:
        return 0
    return angle


def point_line_dist(P0, P1, P2) -> float:
    """Calculate distance between point P0 and a line passing through P1 and P2

    Returns:
        float: distance
    """
    dist = abs(
        (P2[0] - P1[0]) * (P1[1] - P0[1]) - (P1[0] - P0[0]) * (P2[1] - P1[1])
    ) / np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)
    return dist


def point_dist(P0, P1) -> float:
    """Calculate the distance betwoon two points

    Returns:
        float: distance between P0 and P1
    """
    return np.linalg.norm(P0 - P1)


rng = np.random.default_rng(123)
N = 20
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
pointA = leftpoint - np.array([0, 1])
pointB = leftpoint
concave_hull.append(pointB)

# find concave hull
while len(indices) > 1:
    option1 = max(indices, key=lambda pi: calcAngle(pointA, pointB, points[pi]))
    pointC = points[option1]
    winnerangle = calcAngle(pointA, pointB, pointC)
    indices.remove(option1)

    # print(pointA, pointB, pointC)
    option2 = -1
    distance = np.inf
    distBC = point_dist(pointB, pointC)
    for i in indices:
        pi = points[i]
        if i == option1:
            print("i is same as option1")
            continue
        if np.all(np.equal(pointB, pi)):
            print("i is same as pointB")
            continue
        if np.all(np.equal(pointC, pi)):
            print("i is same as pointC")
            continue
        if point_dist(pointB, pi) > distBC:
            # print(f"({pi[0]:.3f}, {pi[1]:.3f}) too far from B")
            continue
        if point_dist(pointC, pi) > distBC:
            # print(f"({pi[0]:.3f}, {pi[1]:.3f}) too far from C")
            continue

        dist_test = point_line_dist(pi, pointB, pointC)
        if dist_test <= distance:
            distance = dist_test
            option2 = i

    if distance < 0.03:
        indices.append(option1)
        indices.remove(option2)
        pointC = points[option2]

    print(f"pointA : ({pointA[0]:<#8.3g}, {pointA[1]:<#8.3g})")
    print(f"pointB : ({pointB[0]:<#8.3g}, {pointB[1]:<#8.3g})")
    print(
        f"option1: ({points[option1][0]:<#8.3g}, {points[option1][1]:<#8.3g}) angle: {180*winnerangle/np.pi:.3g}"
    )
    if option2 >= 0:
        print(
            f"option2: ({points[option2][0]:<#8.3g}, {points[option2][1]:<#8.3g}) dist : {distance:.3g}"
        )
    else:
        print("No valid option 2 found")
    print(f"pointC : ({pointC[0]:<#8.3g}, {pointC[1]:<#8.3g})\n")
    concave_hull.append(pointC)

    # keep point A when making concave turn so hull does not collapse
    # if np.all(np.equal(pointC, points[option2])):
    #     pointB = pointC
    # else:
    pointA, pointB = pointB, pointC

# concave_hull.append(points[indices[0]])
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
