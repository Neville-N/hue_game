from ownFuncs.shape import Shape
import numpy as np
from matplotlib import use as matplotlib_use
matplotlib_use('TkAgg')
import matplotlib.pyplot as plt


def colSpacePlot(Shapes: list[Shape], drawConnections=True, markerSize=20):
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    for color, cur_shape in Shapes.items():
        
        # Draw connecting lines
        if drawConnections:
            for neighbour in cur_shape.neighbours:
                ax.plot([cur_shape.color[2], neighbour.color[2]],
                        [cur_shape.color[1], neighbour.color[1]],
                        [cur_shape.color[0], neighbour.color[0]], color='k', alpha=.5)

        # Draw dots for each shape
        ax.plot(cur_shape.color[2], cur_shape.color[1], cur_shape.color[0],
                marker='o', color=np.flip(np.array(cur_shape.color)/255.0), alpha=1, ms=markerSize)
        
        # Mark locked shapes
        if cur_shape.locked:
            print("locked shape")
            ax.plot(cur_shape.color[2], cur_shape.color[1], cur_shape.color[0],
                marker='x', color='black', alpha=1, ms=markerSize/2)
        
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
    plt.show()
