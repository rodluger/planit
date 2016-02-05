#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
gp2d.py
-------

http://matplotlib.org/basemap/users/installing.html

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import matplotlib.pyplot as pl
import numpy as np
import george
from george.kernels import Matern32Kernel, ConstantKernel, WhiteKernel
from matplotlib.colors import LinearSegmentedColormap
try:
  from mpl_toolkits.basemap import Basemap
except:
  Basemap = None

cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.4, 0.0, 0.0),
                   (0.5, 0.25, 0.25),
                   (0.6, 0.5, 0.5),
                   (1.0, 0.5, 0.5)),

         'green': ((0.0, 0.0, 0.0),
                   (0.4, 0.0, 0.0),
                   (0.5, 0.15, 0.15),
                   (0.6, 0.3, 0.3),
                   (1.0, 0.3, 0.3)),

         'blue':  ((0.0, 0.5, 0.5),
                   (0.4, 1.0, 1.0),
                   (0.5, 0.5, 0.5),
                   (0.6, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
         
blue_brown = LinearSegmentedColormap('BlueBrown', cdict)

size = 50
water_level = 0.5
kernel2d = WhiteKernel(0.005, ndim=2) + ConstantKernel(0.5, ndim=2) * Matern32Kernel(0.5, ndim=2)
kernel1d = WhiteKernel(0.005, ndim=1) + ConstantKernel(0.5, ndim=1) * Matern32Kernel(0.5, ndim=1)

# Our grid coordinates
x = np.linspace(0, 1, num=size)
y = np.linspace(0, 1, num=size)

# Generate the edge @ longitude = +/-180deg
gp = george.GP(kernel1d)
edge = gp.sample(y)
edge = np.append(edge, edge)

# Now sample from the conditional distribution
gp = george.GP(kernel2d)
gp.compute(np.array(np.meshgrid(np.array([0,1]),y)).reshape(2, -1).T, yerr = 0)
z = gp.sample_conditional(edge, np.array(np.meshgrid(x,y)).reshape(2, -1).T)
z -= np.min(z)
z /= np.max(z)

# Plot the contour
pl.contourf(x, y, z.reshape(size,size), levels = np.linspace(0, 1, size), cmap = blue_brown)

# Plot the water level
pl.contour(x, y, z.reshape(size,size), levels = [0, water_level], lw = 2, colors = ['w', 'w'])
pl.xlim(0,1)
pl.ylim(0,1)
pl.show()

# Plot on sphere (TODO)
if False and Basemap is not None:
  map = Basemap(projection='ortho', lat_0 = 50, lon_0 = -100,
                resolution = 'l', area_thresh = 1000.)
  # plot surface
  map.warpimage(image='?')
  # draw the edge of the map projection region (the projection limb)
  map.drawmapboundary()
  # draw lat/lon grid lines every 30 degrees.
  map.drawmeridians(np.arange(0, 360, 30))
  map.drawparallels(np.arange(-90, 90, 30))
  pl.show()