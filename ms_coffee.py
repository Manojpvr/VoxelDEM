# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:13:07 2021

@author: pvrma
"""
"""
This is the script for voxelising the ellipsoidal super quadric particle
data. 
"""

import voxelization
zmax = -0.003 #removes particles above this height, to avoid heaping effect
par_data = voxelization.import_sq_data('sq_ellipsoid_dem_out.csv', zmax)#importing dem output data
rmin = 0.0006 #used to calculate the normalized_voxel_size
normalized_voxel_size = 0.1
bounding_box_dimensions = [-0.0125,0.0125,-0.0125,0.0125,-0.05,-0.001]#[xmin,xmax,ymin,ymax,zmin,zmax]
"""
bounding box dimensions need to be set in such a way that no particle pritudes outside it.
"""
voxel = voxelization.sq_vox(par_data, rmin, bounding_box_dimensions, normalized_voxel_size, 'ellipsoid_sq')#voxelizing the particle data
