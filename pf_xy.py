# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:53:31 2021

@author: pvrma
"""
import numpy as np
from matplotlib import pyplot as plt
import get_pd_dis as gpd
import matplotlib
matplotlib.use('Qt5Agg')

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
normal = [1,0,0]
xmin = -0.0125
xmax = 0.0125
ymin = -0.0125
ymax = 0.0125
zmin = -0.05
zmax = -0.005


x = -0.0125
while x <= 0.0125:
    point = [x,0,-0.03]
    pf = gpd.pd_arb(point,normal,voxel,vd,xmin,xmax,ymin,ymax,zmin,zmax)
    x_list.append([x,pf])
    x += 0.025/200
    print(x,pf)
x_list = np.array(x_list)

y_list = []
normal = [0,1,0]
y = -0.0125
while y <= 0.0125:
    point = [0,y,-0.03]
    pf = gpd.pd_arb(point,normal,voxel,vd,xmin,xmax,ymin,ymax,zmin,zmax)
    y_list.append([y,pf])
    y += 0.025/200
    print(y,pf)
y_list = np.array(y_list)

apf_x = np.sum(x_list[:,1])/np.shape(x_list)[0]
apf_y = np.sum(y_list[:,1])/np.shape(y_list)[0]


np.save('x_pf.npy',x_list)

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

tpfx = round(apf_x,3)
tpfy = round(apf_y,3)

text_size = 30
tick_r = 0.8
plt_xmin = -0.0125
plt_xmax = 0.0125
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 8), dpi=80)
plt.grid(linestyle = 'dotted')
plt.margins(0.01,0.1)
xts = np.arange(plt_xmin,0.0175,0.005)
plt.xlabel('Position(m)',fontsize = text_size)
plt.ylabel('Packing fraction',fontsize = text_size)
plt.plot(x_list[:,0],x_list[:,1],color = 'blue',linewidth = 2)
plt.plot(y_list[:,0],y_list[:,1],color = 'green',linewidth = 2)
plt.legend(['X-direction','Y-direction'], fontsize = text_size)
plt.xticks(xts,fontsize=tick_r*text_size)
plt.yticks(fontsize=tick_r*text_size)
# plt.text(-0.0025, tpfx-0.05, r'$\eta_{a}$ = %f'%tpfx,fontsize = text_size)
plt.savefig("pfd_xy.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')
