# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:57:10 2021

@author: pvrma
"""
import numpy as np
import time
import voxelization

par = np.genfromtxt('coffee_sphere_data.csv', delimiter = ',')#importing multi sphere particle data
"""
In the above line of code, we are importing the multi-sphere configuration for a particle used in our
simulation, i.e, what are the locations of the spheres for a multi-sphere particle located at (0,0,0) and at 
orientation [1,0,0,0]([w,x,y,z]).
"""
x = par[:,0]
y = par[:,1]
z = par[:,2]
r = par[:,3]
n = np.shape(par)[0]
zmax = -0.003
rmin = 0.0006
particle_data = voxelization.import_ms_data('ms_coffee_dem_out.csv',n,x,y,z,r,zmax)
"""
In the above line of code, we are importing the multisphere particles and converting it
into sphere data using the multisphere configuration obtained previously.
"""
normalized_voxel_size = 0.1
bounding_box = [-0.0125,0.0125,-0.0125,0.0125,-0.05,-0.001]
P1 = voxelization.ms_vox(particle_data, rmin, bounding_box, normalized_voxel_size, 'coffee_ms')#voxelizing the sphere data.
