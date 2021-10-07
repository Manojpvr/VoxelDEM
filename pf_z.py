# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:53:31 2021

@author: pvrma
"""
import numpy as np
from matplotlib import pyplot as plt
import get_pd_dis as gpd
import matplotlib
matplotlib.use('agg')

voxel = np.load('voxel.npy')
vd = np.load('voxel_data.npy')

xmin = vd[0]
xmax = vd[1]
ymin = vd[2]
ymax = vd[3]
zmin = vd[4]
zmax = vd[5]


corner_1 = [xmin,ymin,zmin]
corner_2 = [xmin,ymax,zmin]
corner_3 = [xmax,ymax,zmin]
corner_4 = [xmax,ymin,zmin]
corner_5 = [xmin,ymin,zmax]
corner_6 = [xmin,ymax,zmax]
corner_7 = [xmax,ymax,zmax]
corner_8 = [xmax,ymin,zmax]

points = np.array([corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7, corner_8])





x_list = []
normal = [0,0,1]
xmin = -0.0125
xmax = 0.0125
ymin = -0.0125
ymax = 0.0125
zmin = -0.05
zmax = -0.005


z = -0.05
while z <= -0.005:
    point = [0,0,z]
    pf = gpd.pd_arb(point,normal,voxel,vd,xmin,xmax,ymin,ymax,zmin,zmax)
    x_list.append([z,pf])
    z += (0.05-0.003)/200
    print(z,pf)
x_list = np.array(x_list)

total_pf = np.sum(x_list[:,1])/np.shape(x_list)[0]


np.save('z_pf.npy',x_list)

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

total_pf = round(total_pf,3)

text_size = 30
tick_r = 0.8
plt_xmin = -0.05
plt_xmax = 0.0125
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 8), dpi=80)
plt.grid(linestyle = 'dotted')
plt.margins(0.01,0.1)
xts = np.arange(plt_xmin,-0.005+0.009,0.009)
plt.xlabel('Z position(m)',fontsize = text_size)
plt.ylabel('Packing fraction',fontsize = text_size)
plt.plot(x_list[:,0],x_list[:,1],color = 'black',linewidth = 2)
plt.xticks(xts,fontsize=tick_r*text_size)
plt.yticks(fontsize=tick_r*text_size)
plt.text(-0.035, total_pf-0.06, r"$\eta_{a}$ = %.3f"%total_pf,fontsize = text_size)
plt.savefig("pfd_z.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')
