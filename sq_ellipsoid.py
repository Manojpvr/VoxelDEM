# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:57:10 2021

@author: pvrma
"""
import numpy as np
import time
import voxelization

t1 = time.time()
par = np.genfromtxt('coffee_sphere_data.csv', delimiter = ',')
x = par[:,0]
y = par[:,1]
z = par[:,2]
r = par[:,3]
n = np.shape(par)[0]
zmax = -0.003
rmin = 0.0006
particle_data = voxelization.import_ms_data('ms_coffee_dem_out.csv',n,x,y,z,r,zmax)
normalized_voxel_size = 0.1
P1 = voxelization.ms_vox(particle_data,rmin, [-0.0125,0.0125,-0.0125,0.0125,-0.05,-0.001], normalized_voxel_size, 'coffee_ms')
t2 = time.time()

print(t2-t1)
