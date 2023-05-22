import numpy as np
from matplotlib import use as matplotlib_use
import tkinter

matplotlib_use("TkAgg")
# matplotlib_use('Qt5Agg')
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
X = np.sin(t)
Y = np.cos(t)
Z = np.tan(t)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(X, Y, Z)
plt.show()
