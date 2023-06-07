import matplotlib.pyplot as plt
from ownFuncs.shape import Shape
import numpy as np
from matplotlib import use as matplotlib_use
import matplotlib.tri as mtri

matplotlib_use("TkAgg")


def show():
    plt.show(block=True)


def ionshow():
    plt.ion()
    plt.show()


def rgb_space_plot(Shapes: list[Shape], markerSize=20, space="rgb", title="rgb plot"):
    ax = plt.figure(figsize=(12, 12)).add_subplot(projection="3d")
    for shape in Shapes:
        if space.lower() == "rgb":
            location = shape.color
        if space.lower() == "lab":
            location = shape.colorLab
        if space.lower() == "hsv":
            location = shape.colorHsv
            ax.plot(
                location[2],
                location[1],
                location[0] + 255,
                marker="o",
                color=np.flip(np.array(shape.color) / 255.0),
                alpha=1,
                ms=markerSize,
            )

        # Draw dots for each shape
        ax.plot(
            location[2],
            location[1],
            location[0],
            marker="o",
            color=np.flip(np.array(shape.color) / 255.0),
            alpha=1,
            ms=markerSize,
        )

        # Mark locked shapes
        if shape.locked:
            ax.plot(
                location[2],
                location[1],
                location[0],
                marker="x",
                color="black",
                alpha=1,
                ms=markerSize / 2,
            )

    if space.lower() == "rgb":
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
    if space.lower() == "hsv":
        ax.set_xlabel("Value")
        ax.set_ylabel("Saturation")
        ax.set_zlabel("Hue")
    if space.lower() == "lab":
        ax.set_xlabel("B")
        ax.set_ylabel("A")
        ax.set_zlabel("Lightness")

    plt.title(title)
    plt.tight_layout()
    plt.draw()
    plt.pause(1)


def xy_rgb_space_plot(Shapes: list[Shape], markerSize=20, title="xy dots"):
    ax = plt.figure(figsize=(5, 8)).add_subplot()
    for shape in Shapes:
        # Draw dots for each shape
        ax.plot(
            shape.tapX,
            shape.tapY,
            marker="o",
            color=np.flip(np.array(shape.color) / 255.0),
            alpha=1,
            ms=markerSize,
        )

        # Mark locked shapes
        if shape.hardLocked:
            ax.plot(
                shape.tapX,
                shape.tapY,
                marker="x",
                color="black",
                alpha=1,
                ms=markerSize / 2,
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid()
    plt.title(title)
    # plt.axis("equal")
    plt.tight_layout()
    plt.draw()
    plt.pause(1)


def surface_plot(shapes: list[Shape]):
    X = [s.tapX for s in shapes]
    Y = [s.tapY for s in shapes]
    R = [s.color[2] for s in shapes]
    G = [s.color[1] for s in shapes]
    B = [s.color[0] for s in shapes]

    fig = plt.figure(figsize=plt.figaspect(0.3))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(X, Y, R, color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("red")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(X, Y, G, color="green")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("green")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(X, Y, B, color="blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("blue")
    plt.show()


def surfacePlot2(shapes: list[Shape]):
    X = np.array([s.tapX for s in shapes])
    Y = np.array([s.tapY for s in shapes])
    R = np.array([s.color[2] for s in shapes])
    G = np.array([s.color[1] for s in shapes])
    B = np.array([s.color[0] for s in shapes])

    triang = mtri.Triangulation(X, Y)

    fig = plt.figure(figsize=(40, 15))
    # ax = fig.add_subplot(1, 2, 1)

    # ax.triplot(triang_r, c="#D3D3D3", marker='.', markerfacecolor="#DC143C",
    #            markeredgecolor="black", markersize=10)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    labels = ["R", "G", "B"]
    cmaps = ["Reds", "Greens", "Blues"]
    for Z, l, cmap, i in zip([R, G, B], labels, cmaps, range(3)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        ax.plot_trisurf(triang, Z, cmap=cmap)
        ax.scatter(X, Y, Z, marker=".", s=10, c="black", alpha=0.5)
        ax.view_init(elev=60, azim=-45)

        for s in shapes:
            if s.hardLocked:
                ax.plot(
                    s.tapX,
                    s.tapY,
                    s.color[2 - i],
                    marker="o",
                    markersize=20,
                    color="magenta",
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(l)

    plt.show()


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
