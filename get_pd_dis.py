# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:50:15 2021

@author: pvrma
"""

import numpy as np
import math
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt




def dis(x1,y1,z1,x2,y2,z2):
    return(math.sqrt((x1-x2)**2 + (y2-y1)**2 + (z1-z2)**2))

def pd_arb(point,normal,voxel,voxel_data,rv_xmin,rv_xmax,rv_ymin,rv_ymax,rv_zmin,rv_zmax):

    xmin = voxel_data[0]
    xmax = voxel_data[1]
    ymin = voxel_data[2]
    ymax = voxel_data[3]
    zmin = voxel_data[4]
    zmax = voxel_data[5]
    voxel_size = voxel_data[6]


    corner_1 = [xmin,ymin,zmin]
    corner_2 = [xmin,ymax,zmin]
    corner_3 = [xmax,ymax,zmin]
    corner_4 = [xmax,ymin,zmin]
    corner_5 = [xmin,ymin,zmax]
    corner_6 = [xmin,ymax,zmax]
    corner_7 = [xmax,ymax,zmax]
    corner_8 = [xmax,ymin,zmax]

    pts = np.array([corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7, corner_8])

    px = point[0]
    py = point[1]
    pz = point[2]

    point = np.array(point)
    normal = np.array(normal)
    normal = normal/np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2 )

    nx = normal[0]
    ny = normal[1]
    nz = normal[2]

    if rv_xmin < xmin or rv_xmax > xmax or rv_xmin > rv_xmax:
        raise ValueError('Representative volume error')
    if rv_ymin < ymin or rv_ymax > ymax or rv_ymin > rv_ymax:
        raise ValueError('Representative volume error')
    if rv_ymin < ymin or rv_ymax > ymax or rv_ymin > rv_ymax:
        raise ValueError('Representative volume error')
    if px < xmin or px > xmax or py < ymin or py > ymax or pz < zmin or pz > zmax:
        raise ValueError('Hi, I hope you are doing great,the thing is, I want to keep my life simple, so I request you with utmost respect to choose a point inside the domain')



    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [p1x,p1y,p1z] = [(-(ymin-py)*ny-(zmin-pz)*nz)/(nx)+px,ymin,zmin]
    [p2x,p2y,p2z] = [(-(ymin-py)*ny-(zmax-pz)*nz)/(nx)+px,ymin,zmax]
    [p3x,p3y,p3z] = [(-(ymax-py)*ny-(zmax-pz)*nz)/(nx)+px,ymax,zmax]
    [p4x,p4y,p4z] = [(-(ymax-py)*ny-(zmin-pz)*nz)/(nx)+px,ymax,zmin]

    [p5x,p5y,p5z] = [xmin,(-(xmin-px)*nx-(zmin-pz)*nz)/(ny)+py,zmin]
    [p6x,p6y,p6z] = [xmin,(-(xmin-px)*nx-(zmax-pz)*nz)/(ny)+py,zmax]
    [p7x,p7y,p7z] = [xmax,(-(xmax-px)*nx-(zmax-pz)*nz)/(ny)+py,zmax]
    [p8x,p8y,p8z] = [xmax,(-(xmax-px)*nx-(zmin-pz)*nz)/(ny)+py,zmin]

    [p9x,p9y,p9z] = [xmin,ymin,(-(xmin-px)*nx-(ymin-py)*ny)/(nz)+pz]
    [p10x,p10y,p10z] = [xmin,ymax,(-(xmin-px)*nx-(ymax-py)*ny)/(nz)+pz]
    [p11x,p11y,p11z] = [xmax,ymax,(-(xmax-px)*nx-(ymax-py)*ny)/(nz)+pz]
    [p12x,p12y,p12z] = [xmax,ymin,(-(xmax-px)*nx-(ymin-py)*ny)/(nz)+pz]


    # print(p1x,p2x,p3x,p4x,p5x,p6x,p7x,p8x,p9x,p10x,p11x,p12x)
    # print(p1y,p2y,p3y,p4y,p5y,p6y,p7y,p8y,p9y,p10y,p11y,p12y)
    # print(p1z,p2z,p3z,p4z,p5z,p6z,p7z,p8z,p9z,p10z,p11z,p12z)
    if nx == 1 and ny == 0 and nz == 0:

        dis_max = max([dis(px,py,pz,px,ymin,zmin),dis(px,py,pz,px,ymin,zmax),dis(px,py,pz,px,ymax,zmax),dis(px,py,pz,px,ymax,zmin)])

    elif nx == 0 and ny == 1 and nz == 0:

        dis_max = max([dis(px,py,pz,xmin,py,zmin),dis(px,py,pz,xmin,py,zmax),dis(px,py,pz,xmax,py,zmax),dis(px,py,pz,xmax,py,zmin)])

    elif nx == 0 and ny == 0 and nz == 1:

        dis_max = max([dis(px,py,pz,xmin,ymin,pz),dis(px,py,pz,xmin,ymax,pz),dis(px,py,pz,xmax,ymax,pz),dis(px,py,pz,xmax,ymin,pz)])

    else:

        dis_arr = ([dis(px,py,pz,p1x,p1y,p1z),dis(px,py,pz,p2x,p2y,p2z),dis(px,py,pz,p3x,p3y,p3z),dis(px,py,pz,p4x,p4y,p4z),\
                        dis(px,py,pz,p5x,p5y,p5z),dis(px,py,pz,p6x,p6y,p6z),dis(px,py,pz,p7x,p7y,p7z),dis(px,py,pz,p8x,p8y,p8z),\
                            dis(px,py,pz,p9x,p9y,p9z),dis(px,py,pz,p10x,p10y,p10z),dis(px,py,pz,p11x,p11y,p11z),dis(px,py,pz,p12x,p12y,p12z)])

        dis_arr = np.sort(dis_arr)
        dis_max = 1.01*dis_arr[3]
    # print(dis_arr)
    # print(dis_max)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pre_plane_points = []

    pre_plane = np.arange(-dis_max,dis_max,voxel_size)

    n_line = np.shape(pre_plane)[0]

    nplane = n_line**2

    # plane = np.zeros([nplane,4])

    ang1 = math.atan2(ny,nx)
    ang2 = math.atan2(nz,math.sqrt(nx**2+ny**2))
    # print(ang1,ang2)
    # print(point)
    t1 = np.array([[math.cos(ang1),-math.sin(ang1),0],[math.sin(ang1),math.cos(ang1),0],[0,0,1]])
    t2 = np.transpose(np.array([[math.cos(ang2),0,math.sin(ang2)],[0,1,0],[-math.sin(ang2),0,math.cos(ang2)]]))
    T  = np.matmul(t2,t1)


    pre_plane_points = []



    for i in range(n_line):
        for j in range(n_line):
            pre_plane_points.append([0,pre_plane[i],pre_plane[j]])

    pre_plane_points = np.array(pre_plane_points)

    # i = 0
    j = 0

    ip_voxel = []
    while j < nplane:
        pre_point = pre_plane_points[j]
        point_post = point + np.matmul(T, pre_point)

        ppx = point_post[0]
        ppy = point_post[1]
        ppz = point_post[2]
        
        if ppx >= rv_xmin and ppx < rv_xmax and ppy >= rv_ymin and ppy < rv_ymax and ppz >= rv_zmin and ppz < rv_zmax:
            vox_x = int((ppx-xmin)/voxel_size)
            vox_y = int((ppy-ymin)/voxel_size)
            vox_z = int((ppz-zmin)/voxel_size)
            ip_voxel.append((vox_x,vox_y,vox_z))
        j += 1
        

    ip_voxel = list(set(ip_voxel))
    n_tot = len(ip_voxel)
    n_met = 0
    for v  in ip_voxel:
        if voxel[v[0]][v[1]][v[2]] == 1:
            n_met += 1



    pf = n_met/n_tot

    return(pf)
