import matplotlib.pyplot as plt
from ownFuncs.shape import Shape
import numpy as np
from matplotlib import use as matplotlib_use
matplotlib_use('TkAgg')


def colSpacePlot(Shapes: list[Shape], drawConnections=True, markerSize=20):
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    for shape in Shapes:

        # Draw connecting lines
        if drawConnections:
            for neighbour in shape.neighbours:
                ax.plot([shape.color[2], neighbour.color[2]],
                        [shape.color[1], neighbour.color[1]],
                        [shape.color[0], neighbour.color[0]], color='k', alpha=.5)

        # Draw dots for each shape
        ax.plot(shape.color[2], shape.color[1], shape.color[0],
                marker='o', color=np.flip(np.array(shape.color)/255.0), alpha=1, ms=markerSize)

        # Mark locked shapes
        if shape.locked:
            ax.plot(shape.color[2], shape.color[1], shape.color[0],
                    marker='x', color='black', alpha=1, ms=markerSize/2)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    set_axes_equal(ax)


def surfacePlot(shapes: list[Shape]):
    X = [s.centerX for s in shapes]
    Y = [s.centerY for s in shapes]
    R = [s.color[2] for s in shapes]
    G = [s.color[1] for s in shapes]
    B = [s.color[0] for s in shapes]

    fig = plt.figure(figsize=plt.figaspect(0.3))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(X, Y, R, color="red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("red")

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter(X, Y, G, color="green")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("green")

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(X, Y, B, color="blue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("blue")
    plt.show()

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

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
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])