"""
=============================================
Generate polygons to fill under 3D line graph
=============================================

Demonstrate how to create polygons which fill the space under a line
graph. In this example polygons are semi-transparent, creating a sort
of 'jagged stained glass' effect.
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

xs = np.arange(0, 10, 0.4)
verts = []
zs = [0.0, 1.0, 2.0, 3.0]
# print(xs, len(xs))
# exit(0)
for idx,z in enumerate(zs):
    ys = np.power(xs, -1-(idx*0.05))
    # print(ys, len(ys))
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))


poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
                                         cc('y')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('Distance')
ax.set_xlim3d(0, 10)
# ax.set_ylabel('Phones')
ax.set_ylim3d(-1, 4)
ax.set_zlabel('Success Rate %')
ax.set_zlim3d(0, 3)
x_labels = [item.get_text() for item in ax.get_xticklabels()]
# print(labels)
x_labels[0] = '10cm'
x_labels[1] = '20cm'
x_labels[2] = '50cm'
x_labels[3] = '1M'
x_labels[4] = '2M'
x_labels[5] = '10M'
ax.set_xticklabels(x_labels)

y_labels = [item.get_text() for item in ax.get_yticklabels()]
# print(labels)
y_labels[0] = ''
y_labels[1] = 'Samsung Note10'
y_labels[2] = 'Huawei Honor 10'
y_labels[3] = 'Moto G5'
y_labels[4] = 'Xiaomi 8'
y_labels[5] = 'Pixel 3'
print(y_labels)
ax.set_yticklabels(y_labels)

z_labels = [item.get_text() for item in ax.get_zticklabels()]
for idx, la in enumerate(z_labels):
    z_labels[idx] = str(round(float(idx) * 100/6))
print(z_labels)
ax.set_zticklabels(z_labels)



plt.show()
