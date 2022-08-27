# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:13:07 2021

@author: pvrma
"""
import voxelization

xmin_domain = -0.0125
xmax_domain = 0.0125
ymin_domain = -0.0125
ymax_domain = 0.0125
zmin_domain = -0.05
zmax_domain = -0.001

domain_extents = [xmin_domain,xmax_domain,ymin_domain,ymax_domain,zmin_domain,zmax_domain]
zmax = -0.003

"""
zmax refers the maximum height till which the particles to which we need the particles
to be voxelized, this is lesser the Zmax to avoid the heaping effect, if you do not wish
to have it, you can make it equal to zmax_domain
"""


par_data = voxelization.import_sq_data('out.csv', zmax)

rmin = 0.0006
"""
rmin is the smallest particle dimension in your granular assembly, for superquadric,
we have A,B,C, i.e, scaling parameters, rmin will be the smallest of A, B, C. If particles
have different scaling parametes then rmin will be smallest of all of them, you can
either manually enter it, or find it directly from your particle data.
"""


normalised_voxel_size = 0.5
"""
normalied voxel size is ratio of the size of the voxel to rmin

i.e, voxel size = rmin*normalised_voxel_size
"""

voxel = voxelization.sq_vox(par_data,rmin,domain_extents,normalised_voxel_size,'out_sample_name')
