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
normal = [1,1,0]
xmin = -0.0125
xmax = 0.0125
ymin = -0.0125
ymax = 0.0125
zmin = -0.05
zmax = -0.005


x = -0.012
y = -0.012
while x <= 0.012:
    point = [x,y,(zmin+zmax)/2]
    pf = gpd.pd_arb(point,normal,voxel,vd,xmin,xmax,ymin,ymax,zmin,zmax)
    diag_dis = np.sqrt((x+0.012)**2 + (y+0.012)**2)
    x_list.append([diag_dis,pf])
    x += (0.024-0.003)/200
    y += (0.024-0.003)/200
    print(diag_dis,pf)
x_list = np.array(x_list)

total_pf = np.sum(x_list[:,1])/np.shape(x_list)[0]


np.save('diag_pf.npy',x_list)

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

total_pf = round(total_pf,3)

text_size = 24
tick_r = 0.8
plt_xmin = 0
plt_xmax = x_list[-1,0]
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 8), dpi=80)
plt.grid(linestyle = 'dotted')
plt.margins(0.01,0.1)
xts = np.arange(plt_xmin,plt_xmax+0.007,0.007)
plt.xlabel('Plane displacement(m)',fontsize = text_size)
plt.ylabel('Packing fraction',fontsize = text_size)
plt.plot(x_list[:,0],x_list[:,1],color = 'black',linewidth = 2)
plt.xticks(xts,fontsize=tick_r*text_size,weight='bold')
plt.yticks(fontsize=tick_r*text_size,weight='bold')
# plt.text((xmax+xmin)/2, total_pf-0.06, r'$\eta_{a}$ = %f'%total_pf,fontsize = text_size)
plt.savefig("pfd_diag.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')
