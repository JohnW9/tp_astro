import tp_astro as tp
from miscmath import fit_circle
import matplotlib.pyplot as plt
import numpy as np

cam, pos = tp.tp_init()
xData = []
yData = []
traj = [(0,80), (0,160), (0,240)]
for position in traj:
    pos.goto_absolute(position[0], position[1])
    pos.wait_move()
    x,y = cam.getCentroid()
    xData.append(x)
    yData.append(y)

print(xData, yData)
xData = np.asarray(xData)
yData = np.asarray(yData)
cx, cy, r = fit_circle(xData, yData)

plt.scatter(xData, yData)
plt.scatter(cx, cy)