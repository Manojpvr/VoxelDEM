# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:53:31 2021

@author: pvrma
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

voxel = np.load('voxel.npy')
vd = np.load('voxel_data.npy')

zsurf = np.array([[np.sum(voxel[i][j][:])/np.shape(voxel)[2] for i in range(np.shape(voxel)[0])] for j in range(np.shape(voxel)[1])])

ysurf = np.zeros([np.shape(voxel)[0],np.shape(voxel)[2]-40])
for i in range(np.shape(voxel)[0]):
    for j in range(np.shape(voxel)[2]-40):
        a = voxel[i]
        ysurf[i][j] = np.sum(a[:,j])/np.shape(voxel)[1]

textsize = 24
plt.figure()
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"
plt.imshow(np.flipud(np.transpose(zsurf)), interpolation='none', cmap = 'plasma', vmin = 0.25)
plt.axis('off')
plt.arrow(-10, np.shape(voxel)[0]+10, 100, 0, width = 3, color = 'black')
plt.text(35, np.shape(voxel)[0]+35,'x',fontsize = textsize)
plt.arrow(-10, np.shape(voxel)[0]+10, 0, -100, width = 3, color = 'black')
plt.text(-35, np.shape(voxel)[0]-30,'y',fontsize = textsize)
plt.show()
plt.savefig("zsurf.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')

plt.figure()
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"
plt.imshow(np.flipud(np.transpose(ysurf)), interpolation='none', cmap = 'plasma', vmin = 0.25)
plt.axis('off')
plt.arrow(-10, np.shape(voxel)[2]-30, 100, 0, width = 3, color = 'black')
plt.text(35, np.shape(voxel)[2]+20,'x',fontsize = textsize)
plt.arrow(-10, np.shape(voxel)[2]-30, 0, -100, width = 3, color = 'black')
plt.text(-60, np.shape(voxel)[2]-70,'z',fontsize = textsize)
plt.show()
plt.savefig("ysurf.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')
