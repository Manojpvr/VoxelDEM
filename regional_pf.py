# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:53:31 2021

@author: pvrma
"""
import numpy as np
from matplotlib import pyplot as plt
import get_pd_dis as gpd
import matplotlib
import matplotlib.patches as mpatches

matplotlib.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams["font.family"] = "Times New Roman"
plt.figure()
matplotlib.use('Qt5Agg')

plt.text(35, -60,'x',fontsize = 24)

plt.text(-60, 30,'z',fontsize = 24)
plt.arrow(-10, -10, 100, 0, width = 3, color = 'black')
plt.arrow(-10, -10, 0, 100, width = 3, color = 'black')
plt.show()


voxel = np.load('voxel.npy')
vd = np.load('voxel_data.npy')

nx = np.shape(voxel)[0]
ny = np.shape(voxel)[1]
nz = np.shape(voxel)[2]-40

xmind = vd[0]
xmaxd = vd[1]
ymind = vd[2]
ymaxd = vd[3]
zmind = vd[4]
zmaxd = vd[5]

xmin = 0
xmax = nx
ymin = 0
ymax = ny
zmin = 0
zmax = nz



corner_1 = [xmin,ymin,zmin]
corner_2 = [xmin,ymax,zmin]
corner_3 = [xmax,ymax,zmin]
corner_4 = [xmax,ymin,zmin]
corner_5 = [xmin,ymin,zmax]
corner_6 = [xmin,ymax,zmax]
corner_7 = [xmax,ymax,zmax]
corner_8 = [xmax,ymin,zmax]

points = np.array([corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7, corner_8])

vnx = np.shape(voxel)[0]
vny = np.shape(voxel)[1]
vnz = np.shape(voxel)[2]-40



zrmin = int(vnz/4)
zrmax = vnz - zrmin
zrarr1 = np.arange(0,zrmin,1)
zrarr2 = np.arange(zrmin,zrmax,1)
zrarr3 = np.arange(zrmax,vnz,1)

xrmin = int(vnx/4)
xrmax = vnx - xrmin
xrarr1 = np.arange(0,xrmin,1)
xrarr2 = np.arange(xrmin,xrmax,1)
xrarr3 = np.arange(xrmax,vnx,1)

r1 = voxel[xrarr1,:,:]
r1 = r1[:,:,zrarr1]
nvr1 = np.shape(r1)[0] * np.shape(r1)[1] *  np.shape(r1)[2]
nmr1 = np.sum(r1)
pdr1 = nmr1/nvr1
print(pdr1)

r2 = voxel[xrarr2,:,:]
r2 = r2[:,:,zrarr1]
nvr2 = np.shape(r2)[0] * np.shape(r2)[1] *  np.shape(r2)[2]
nmr2 = np.sum(r2)
pdr2 = nmr2/nvr2
print(pdr2)

r3 = voxel[xrarr3,:,:]
r3 = r3[:,:,zrarr1]
nvr3 = np.shape(r3)[0] * np.shape(r3)[1] *  np.shape(r3)[2]
nmr3 = np.sum(r3)
pdr3 = nmr3/nvr3
print(pdr3)

r4 = voxel[xrarr3,:,:]
r4 = r4[:,:,zrarr2]
nvr4 = np.shape(r4)[0] * np.shape(r4)[1] *  np.shape(r4)[2]
nmr4 = np.sum(r4)
pdr4 = nmr4/nvr4
print(pdr4)

r5 = voxel[xrarr3,:,:]
r5 = r5[:,:,zrarr3]
nvr5 = np.shape(r5)[0] * np.shape(r5)[1] *  np.shape(r5)[2]
nmr5 = np.sum(r5)
pdr5 = nmr5/nvr5
print(pdr5)

r6 = voxel[xrarr2,:,:]
r6 = r6[:,:,zrarr3]
nvr6 = np.shape(r6)[0] * np.shape(r6)[1] *  np.shape(r6)[2]
nmr6 = np.sum(r6)
pdr6 = nmr6/nvr6
print(pdr6)

r7 = voxel[xrarr1,:,:]
r7 = r7[:,:,zrarr3]
nvr7 = np.shape(r7)[0] * np.shape(r7)[1] *  np.shape(r7)[2]
nmr7 = np.sum(r7)
pdr7 = nmr7/nvr7
print(pdr7)

r8 = voxel[xrarr1,:,:]
r8 = r8[:,:,zrarr2]
nvr8 = np.shape(r8)[0] * np.shape(r8)[1] *  np.shape(r8)[2]
nmr8 = np.sum(r8)
pdr8 = nmr8/nvr8
print(pdr8)

r9 = voxel[xrarr2,:,:]
r9 = r9[:,:,zrarr2]
nvr9 = np.shape(r9)[0] * np.shape(r9)[1] *  np.shape(r9)[2]
nmr9 = np.sum(r9)
pdr9 = nmr9/nvr9
print(pdr9)












text_size = 15
ax = plt.gca()


p1 = mpatches.Rectangle((xmin,(zmin)),(xmax-xmin)/4, (zmax-zmin)/4,color="lightgreen")
ax.text(xmin + (xmax-xmin)/200, zmin + (zmax-zmin)/8, r"%.3f"%pdr1,fontsize = text_size, weight='bold')

p2 = mpatches.Rectangle((xmin+(xmax-xmin)/4,zmin),(xmax-xmin)/2, (zmax-zmin)/4,color="lightblue")
ax.text(xmin + (xmax-xmin)/2.6, zmin + (zmax-zmin)/8, r"%.3f"%pdr2,fontsize = text_size, weight='bold')

p3 = mpatches.Rectangle((xmin+3*(xmax-xmin)/4,zmin),(xmax-xmin)/4, (zmax-zmin)/4,color="lightgreen")
ax.text(xmin + 3*(xmax-xmin)/4, zmin + (zmax-zmin)/8, r"%.3f"%pdr3,fontsize = text_size, weight='bold')

p4 = mpatches.Rectangle((xmin+3*(xmax-xmin)/4,zmin+(zmax-zmin)/4),(xmax-xmin)/4, (zmax-zmin)/2,color="lightblue")
ax.text(xmin + 3*(xmax-xmin)/4, zmin + (zmax-zmin)/2, r"%.3f"%pdr4,fontsize = text_size, weight='bold')

p5 = mpatches.Rectangle((xmin+3*(xmax-xmin)/4,zmin+3*(zmax-zmin)/4),(xmax-xmin)/4, (zmax-zmin)/4,color="lightgreen")
ax.text(xmin + 3*(xmax-xmin)/4, zmin + 7*(zmax-zmin)/8, r"%.3f"%pdr5,fontsize = text_size, weight='bold')

p6 = mpatches.Rectangle((xmin+(xmax-xmin)/4,zmin+3*(zmax-zmin)/4),(xmax-xmin)/2, (zmax-zmin)/4,color="lightblue")
ax.text(xmin + (xmax-xmin)/2.6, zmin + 7*(zmax-zmin)/8, r"%.3f"%pdr6,fontsize = text_size, weight='bold')

p7 = mpatches.Rectangle((xmin,zmin+3*(zmax-zmin)/4),(xmax-xmin)/4, (zmax-zmin)/4,color="lightgreen")
ax.text(xmin + (xmax-xmin)/200, zmin + 7*(zmax-zmin)/8, r"%.3f"%pdr7,fontsize = text_size, weight='bold')

p8 = mpatches.Rectangle((xmin,zmin+(zmax-zmin)/4),(xmax-xmin)/4, (zmax-zmin)/2,color="lightblue")
ax.text(xmin + (xmax-xmin)/200, zmin + (zmax-zmin)/2, r"%.3f"%pdr8,fontsize = text_size, weight='bold')

p9 = mpatches.Rectangle((xmin+(xmax-xmin)/4,zmin+(zmax-zmin)/4),(xmax-xmin)/2, (zmax-zmin)/2,color="lightcoral")
ax.text(xmin + (xmax-xmin)/2.6, zmin + (zmax-zmin)/2, r"%.3f"%pdr9,fontsize = text_size, weight='bold')


p1.set_clip_on(False)
p2.set_clip_on(False)
p3.set_clip_on(False)
p4.set_clip_on(False)
p5.set_clip_on(False)
p6.set_clip_on(False)
p7.set_clip_on(False)
p8.set_clip_on(False)
p9.set_clip_on(False)

p_bb = mpatches.Rectangle((xmin-0.001,zmin-0.001), xmax-xmin+0.002, zmax-zmin+0.002, color = 'white')
p_bb.set_clip_on(False)
ax.add_patch(p_bb)

ax.add_patch(p1)
ax.add_patch(p2)
ax.add_patch(p3)
ax.add_patch(p4)
ax.add_patch(p5)
ax.add_patch(p6)
ax.add_patch(p7)
ax.add_patch(p8)
ax.add_patch(p9)



# plt.axvline(-0.0125)
# plt.axhline(-0.05)
ax.set_aspect('equal')
# plt.xlim(xmin,xmax)
# plt.ylim(zmin,zmax)
plt.margins(0,0)
plt.axis('off')
plt.savefig("ref.pdf", dpi=600, pad_inches = 0, bbox_inches='tight')
plt.show()