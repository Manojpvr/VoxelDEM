
"""
Created on Sun Jul 18 08:39:44 2021
@author: pvrma
"""
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from time import time as time

def voxelize_sq_par(par_x,par_y,par_z,q0,q1,q2,q3,a,b,c,n1,n2,voxel_grid,voxel_size,xmin,xmax,ymin,ymax,zmin,zmax):

    r = R.from_quat([q0,q1,q2,q3])
    T = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],[2*(q1*q2 - q0*q3), 1-(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],[2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1**2 + q2**2)]])
    T = r.as_matrix()
    rbs = max(a,b,c) #radius of bounding sphere

    xpi = par_x - rbs - voxel_size
    ypi = par_y - rbs - voxel_size
    zpi = par_z - rbs - voxel_size


    xpf = par_x + rbs + voxel_size
    ypf = par_y + rbs + voxel_size
    zpf = par_z + rbs + voxel_size

    xvox = math.floor((xpi-xmin)/voxel_size)
    if xvox < 0:
        xvox = 0
    yvox = math.floor((ypi-ymin)/voxel_size)
    if yvox < 0:
        yvox = 0
    zvox = math.floor((zpi-zmin)/voxel_size)
    if zvox < 0:
        zvox = 0
    xvoxf = math.floor((xpf-xmin)/voxel_size)
    if xvoxf > np.shape(voxel_grid)[0] - 1:
        xvoxf =  np.shape(voxel_grid)[0] - 1
    yvoxf = math.floor((ypf-ymin)/voxel_size)
    if yvoxf > np.shape(voxel_grid)[1] - 1:
        yvoxf =  np.shape(voxel_grid)[1] - 1
    zvoxf = math.floor((zpf-zmin)/voxel_size)
    if zvoxf > np.shape(voxel_grid)[2] - 1:
        zvoxf =  np.shape(voxel_grid)[2] - 1


    xp = (xvox+0.5)*voxel_size+xmin
    yp = (yvox+0.5)*voxel_size+ymin
    zp = (zvox+0.5)*voxel_size+zmin
    while xvox <= xvoxf:
        yvox = math.floor((ypi-ymin)/voxel_size)
        if yvoxf > np.shape(voxel_grid)[1] - 1:
            yvoxf =  np.shape(voxel_grid)[1] - 1
        yp = (yvox+0.5)*voxel_size+ymin
        while yvox <= yvoxf:
            zvox = math.floor((zpi-zmin)/voxel_size)
            if zvoxf > np.shape(voxel_grid)[2] - 1:
                zvoxf =  np.shape(voxel_grid)[2] - 1
            zp = (zvox+0.5)*voxel_size+zmin

            while zvox <= zvoxf:
                    if voxel_grid[xvox,yvox,zvox] == 0:

                            p_translate = np.array([xp,yp,zp]) - np.array([par_x,par_y,par_z])

                            p_global = np.matmul(T.transpose(),p_translate.transpose())

                            p_glob_x = p_global[0]
                            p_glob_y = p_global[1]
                            p_glob_z = p_global[2]


                            p_val = (((math.fabs(p_glob_x/a))**n2) + ((math.fabs(p_glob_y/b))**n2))**(n1/n2) + (math.fabs(p_glob_z/c))**n1 - 1

                            if p_val <= 0:
                                voxel_grid[xvox,yvox,zvox] = 1

                    zvox += 1
                    zp += voxel_size

            yvox += 1
            yp += voxel_size

        xvox += 1
        xp+= voxel_size

    return(voxel_grid)

def voxelize_ms_par(par_x,par_y,par_z,r,voxel_grid,voxel_size,xmin,xmax,ymin,ymax,zmin,zmax):

    xini = par_x - 0.55*r
    yini = par_y - 0.55*r
    zini = par_z - 0.55*r

    xinf = par_x + 0.55*r
    yinf = par_y + 0.55*r
    zinf = par_z + 0.55*r

    xivox = math.floor((xini-xmin)/voxel_size)
    yivox = math.floor((yini-ymin)/voxel_size)
    zivox = math.floor((zini-zmin)/voxel_size)

    xfvox = math.floor((xinf-xmin)/voxel_size)
    yfvox = math.floor((yinf-ymin)/voxel_size)
    zfvox = math.floor((zinf-zmin)/voxel_size)


    voxel_grid[xivox:xfvox,yivox:yfvox,zivox:zfvox] = [[[1 for i in range(zfvox-zivox)]for j in range(yfvox-yivox)]for k in range(xfvox-xivox)]


    xpi = par_x - r - voxel_size
    ypi = par_y - r - voxel_size
    zpi = par_z - r - voxel_size

    xpf = par_x + r + voxel_size
    ypf = par_y + r + voxel_size
    zpf = par_z + r + voxel_size

    xvox = math.floor((xpi-xmin)/voxel_size)
    if xvox < 0:
        xvox = 0
    yvox = math.floor((ypi-ymin)/voxel_size)
    if yvox < 0:
        yvox = 0
    zvox = math.floor((zpi-zmin)/voxel_size)
    if zvox < 0:
        zvox = 0
    xvoxf = math.floor((xpf-xmin)/voxel_size)
    if xvoxf > np.shape(voxel_grid)[0] - 1:
        xvoxf =  np.shape(voxel_grid)[0] - 1
    yvoxf = math.floor((ypf-ymin)/voxel_size)
    if yvoxf > np.shape(voxel_grid)[1] - 1:
        yvoxf =  np.shape(voxel_grid)[1] - 1
    zvoxf = math.floor((zpf-zmin)/voxel_size)
    if zvoxf > np.shape(voxel_grid)[2] - 1:
        zvoxf =  np.shape(voxel_grid)[2] - 1


    xp = (xvox+0.5)*voxel_size+xmin
    yp = (yvox+0.5)*voxel_size+ymin
    zp = (zvox+0.5)*voxel_size+zmin
    while xvox <= xvoxf:
        yvox = math.floor((ypi-ymin)/voxel_size)
        if yvoxf > np.shape(voxel_grid)[1] - 1:
            yvoxf =  np.shape(voxel_grid)[1] - 1
        yp = (yvox+0.5)*voxel_size+ymin
        while yvox <= yvoxf:
            zvox = math.floor((zpi-zmin)/voxel_size)
            if zvoxf > np.shape(voxel_grid)[2] - 1:
                zvoxf =  np.shape(voxel_grid)[2] - 1
            zp = (zvox+0.5)*voxel_size+zmin

            while zvox <= zvoxf:
                    if voxel_grid[xvox,yvox,zvox] == 0:

                            p_translate = np.array([xp,yp,zp]) - np.array([par_x,par_y,par_z])

                            p_glob_x = p_translate[0]
                            p_glob_y = p_translate[1]
                            p_glob_z = p_translate[2]


                            p_val = ((p_glob_x/r)**2) + ((p_glob_y/r)**2) + ((p_glob_z/r)**2) - 1

                            if p_val <= 0:
                                voxel_grid[xvox,yvox,zvox] = 1

                    zvox += 1
                    zp += voxel_size

            yvox += 1
            yp += voxel_size

        xvox += 1
        xp+= voxel_size

    return(voxel_grid)

def prepend_line(file_name, line):
    #Insert given string as a new line at the beginning of a file
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

def save_to_csv(sqd, n):
    np.savetxt('sq_data.xyz', sqd, delimiter=" ")
    prepend_line('sq_data.xyz', '"particle data"')
    prepend_line('sq_data.xyz', str(n))
    return()

def import_sq_data(filename,zmax):
    sq_data = np.genfromtxt(filename,delimiter = ',')
    n = np.shape(sq_data)[0]
    a = []
    for i in range(n):
        sq_data[i][10] = 2/sq_data[i][10]
        sq_data[i][11] = 2/sq_data[i][11]
        if sq_data[i][2] > zmax:
            a.append(i)

    sq_data = np.delete(sq_data,a,0)
    n = np.shape(sq_data)[0]
    save_to_csv(sq_data, n)
    return(sq_data)

def import_ms_data(filename,ms_config_data_file,zmax):
    ti = time()
    ms_config_data = np.genfromtxt(ms_config_data_file, delimiter = ',')
    n = np.shape(ms_config_data)[0]
    x = ms_config_data[:,0]
    y = ms_config_data[:,1]
    z = ms_config_data[:,2]
    r = ms_config_data[:,3]
    a = np.genfromtxt(filename, delimiter = ',')
    a = np.transpose(a)
    N = np.shape(a)[0]
    rem = []
    for i in range(N):
        if a[i][2] > zmax:
            rem.append(i)
    a = np.delete(a,rem,0)
    N = np.shape(a)[0]
    P = np.zeros([N*n,4])
    for i in range(N):
        for j in range(n):
            P[n*i+j][3] = r[j]
            R = [[a[i][3],a[i][4],a[i][5]],[a[i][6],a[i][7],a[i][8]],[a[i][9],a[i][10],a[i][11]]]
            xlb = x[j]
            ylb = y[j]
            zlb = z[j]
            lb = [xlb,ylb,zlb]
            ab = np.matmul((R),np.transpose(lb))
            P[n*i+j][0] = a[i][0]+ab[0]
            P[n*i+j][1] = a[i][1]+ab[1]
            P[n*i+j][2] = a[i][2]+ab[2]
    np.savetxt('Sphere.csv', P, delimiter=", ")
    tf = time()
    time_import = tf-ti
    print('DEM multisphere data imported in %f seconds'%time_import)
    return(P)

def sq_vox(sq_data,rmin,domain_data,normalized_voxel_size,out_file_name):
    n = np.shape(sq_data)[0]
    xmin = domain_data[0]
    xmax = domain_data[1]
    ymin = domain_data[2]
    ymax = domain_data[3]
    zmin = domain_data[4]
    zmax = domain_data[5]
    voxel_size = rmin*normalized_voxel_size
    nx = math.ceil((xmax-xmin)/voxel_size)
    ny = math.ceil((ymax-ymin)/voxel_size)
    nz = math.ceil((zmax-zmin)/voxel_size)
    voxel_array = np.zeros([nx,ny,nz])
    for i in tqdm(range(n), desc="Voxelizing the particle data"):
        voxel_array = voxelize_sq_par(sq_data[i][0],sq_data[i][1],sq_data[i][2],sq_data[i][3],sq_data[i][4],sq_data[i][5],sq_data[i][6],sq_data[i][7],sq_data[i][8],sq_data[i][9],2/(sq_data[i][10]),2/(sq_data[i][11]),voxel_array,voxel_size,xmin,xmax,ymin,ymax,zmin,zmax)
    voxel_out_name = out_file_name+'.npy'
    np.save(voxel_out_name,voxel_array)
    voxel_data_out_name = out_file_name+'_vdata.npy'
    voxel_data = np.array([xmin,xmax,ymin,ymax,zmin,zmax,voxel_size])
    np.save(voxel_data_out_name,voxel_data)
    return(voxel_array)

def ms_vox(ms_data,rmin,domain_data,normalized_voxel_size,out_file_name):
    n = np.shape(ms_data)[0]
    xmin = domain_data[0]
    xmax = domain_data[1]
    ymin = domain_data[2]
    ymax = domain_data[3]
    zmin = domain_data[4]
    zmax = domain_data[5]

    voxel_size = rmin*normalized_voxel_size
    nx = math.ceil((xmax-xmin)/voxel_size)
    ny = math.ceil((ymax-ymin)/voxel_size)
    nz = math.ceil((zmax-zmin)/voxel_size)
    voxel_array = np.zeros([nx,ny,nz])
    for i in tqdm(range(n), desc="Voxelizing the particle data"):
    # for i in range(n):
        voxel_array = voxelize_ms_par(ms_data[i][0],ms_data[i][1],ms_data[i][2],ms_data[i][3],voxel_array,voxel_size,xmin,xmax,ymin,ymax,zmin,zmax)
    voxel_out_name = out_file_name+'.npy'
    np.save(voxel_out_name,voxel_array)
    voxel_data_out_name = out_file_name+'_vdata.npy'
    voxel_data = np.array([xmin,xmax,ymin,ymax,zmin,zmax,voxel_size])
    np.save(voxel_data_out_name,voxel_data)
    return(voxel_array)

def voxel_plot(voxel):
    matplotlib.use('agg')

    # fig2 = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, alpha = 0.9)
    # plt.plot()
    plt.savefig("voxel_plot", dpi=500)
    return()
