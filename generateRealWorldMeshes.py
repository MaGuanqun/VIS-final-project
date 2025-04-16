from netCDF4 import Dataset
import numpy as np
import random
import numpy.random as npr
from pyevtk.hl import *
from math import e
from unionFind import *
import os
import torch as th
import gzip
from scipy.ndimage import generic_filter

random.seed(0)
np.random.seed(0)

rel_pos = {
    0 : (0,-1),
    1 : (1,-1),
    2 : (1,0),
    3 : (0,1),
    4 : (-1,1),
    5 : (-1,0),
}

# adjacency = {
#     0: [1],
#     1: [2],
#     2: [3],
#     3: [4,5],
#     4: [5],
#     5: [6],
#     6: [7],
#     7: [1]
# }

adjacency = {
    0: [1,5],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [],
}

def classifyPoint(points, p):
    w,h = points.shape

    if p[0] == 0:
        if p[1] == 0:
            valid_neighbors = [2,3]
            num_ignored = 4
        elif p[1] == h-1:
            valid_neighbors = [0,1,2]
            num_ignored = 3
        else:
            valid_neighbors = [0,1,2,3]
            num_ignored = 2
    elif p[0] == w-1:
        if p[1] == 0:
            valid_neighbors = [3,4,5]
            num_ignored = 3
        elif p[1] == h-1:
            valid_neighbors = [0,5]
            num_ignored = 4
        else:
            valid_neighbors = [0,3,4,5]
            num_ignored = 2
    else:
        if p[1] == 0:
            valid_neighbors = [2,3,4,5]
            num_ignored = 2
        elif p[1] == h-1:
            valid_neighbors = [0,1,2,5]
            num_ignored = 2
        else:
            valid_neighbors = [0,1,2,3,4,5]
            num_ignored = 0
    
    assignments = [0] * 6

    possible_min = True

    val = points[p[0],p[1]]

    for n in valid_neighbors:
        off = rel_pos[n]

        otherval = points[p[0] + off[0], p[1] + off[1]]

        if val == otherval:
            if off[1] == -1 or (off[1] == 0 and off[0] == -1):
                assignments[n] = 1
                possible_min = False
            else:
                assignments[n] = 2
        elif otherval < val:
            assignments[n] = 1
            possible_min = False
        else:
            assignments[n] = 2

    uf = UnionFind(6)

    for n in valid_neighbors:
        for n2 in adjacency[n]:
            if assignments[n] == assignments[n2]:
                uf.union(n,n2)
    
    cc = uf.n_components - num_ignored

    if cc == 1:
        if possible_min:
            return 0
        else:
            return 2
    if cc == 2:
        return 3
    else:
        return 1


def load_data(file_path,i):
    nc_data = Dataset(file_path, mode='r')
    print(nc_data.variables['u'].shape)

    u1 = nc_data.variables['u'][i, :, :].filled(-2.0)  # Shape: (tdim, ydim, xdim)
    v1 = nc_data.variables['v'][i, :, :].filled(-2.0)
    nc_data.close()


    velocity_t1501 = np.sqrt(u1**2 + v1**2)
    # velocity_t1501 = np.expand_dims(velocity_t1501, axis=2)
    out = (velocity_t1501 - np.min(velocity_t1501)) / (np.max(velocity_t1501) - np.min(velocity_t1501))
    print(out)
    
    return out


def load_hurricane_isabel(file_path,i):
    XDIM = 500
    YDIM = 500
    ZDIM = 100
    TDIM = 1  # Single time step for each file

    # Load the binary data
    with gzip.open(file_path, 'rb') as f:
        # Step 2: Read the data as Big Endian float32
        data = np.frombuffer(f.read(), dtype='>f4')

    expected_size = XDIM * YDIM * ZDIM * TDIM
    if data.size != expected_size:
        raise ValueError(f"Data size {data.size} does not match expected size {expected_size}.")
    data = data.reshape((TDIM,ZDIM,YDIM, XDIM))
    slice_z50 = data[0, i, :, :].copy()  

    missing_value = 1e35
    slice_z50[slice_z50 == missing_value] = np.nan

    def nanmean_filter(values):
        valid_values = values[~np.isnan(values)]
        if valid_values.size > 0:
            return np.mean(valid_values)
        else:
            return np.nan
        
    while np.isnan(slice_z50).any():
        slice_z50 = generic_filter(slice_z50, nanmean_filter, size=3, mode='constant', cval=np.nan)

    slice_z50 = np.nan_to_num(slice_z50, nan=missing_value)
    
    out = (slice_z50 - np.min(slice_z50)) / (np.max(slice_z50) - np.min(slice_z50))
    print(out)
    
    return out
    


def load_bin(file_path):
    XDIM = 3600
    YDIM = 1800
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape((YDIM, XDIM))
    out = (data - np.min(data)) / (np.max(data) - np.min(data))
    print(out)
    return out


if __name__ == "__main__":

    data_dir = "./real_data"

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    
    file_path=[]
    file_path.append('./ori/boussinesq.nc')
    file_path.append('./ori/cylinder2d.nc')
    file_path.append('./ori/Pf30.bin.gz')
    file_path.append('./ori/FLDSC_1_1800_3600.dat')
    
    num=50+50+20+2+2+1+1+1+1+1
    
    for i in range(num):
        if(i<50):
            out = load_data(file_path[0],i+1000)        
        elif(i<100):
            out = load_data(file_path[1],i-50+1000)
        elif(i<120):
            out = load_hurricane_isabel(file_path[2],i-120)
        elif(i<122):
            out =load_data(file_path[0],i-120+1050)
        elif(i<124):
            out = load_data(file_path[1],i-122+1050)
        elif(i<125):
            out = load_hurricane_isabel(file_path[2],i-124+20)
        elif(i<126): 
            out =load_data(file_path[0],i-125+1052)
        elif(i<127):
            out = load_data(file_path[1],i-126+1052)
        elif(i<128):
            out = load_hurricane_isabel(file_path[2],i-127+21)
        else:
            out = load_bin(file_path[3]) 
        idx = i
        print(idx)
        w,h= out.shape[0], out.shape[1]
        out_i = np.zeros((w,h))
        out_j = np.zeros((w,h))
        for i in range(w):
            for j in range(h):
                out_i[i,j] = i
                out_j[i,j] = j

        labels = np.zeros((w,h),dtype=np.int64)
        for i in range(w):
            for j in range(h):
                labels[i,j] = classifyPoint(out, (i,j))

        min_out = np.min(out)
        max_out = np.max(out)
        out = (out - min_out) / (max_out - min_out)

        # out = out.T.reshape(-1)
        # out_i = out_i.T.reshape(-1)
        # out_j = out_j.T.reshape(-1)

        # labels = labels.T.reshape(-1)

        mesh_e1 = []
        mesh_e2 = []

        for i in range(w):
            for j in range(h):
                idx1 = j*w+i

                if i != w-1:
                    idx2 = idx1+1

                    mesh_e1.append(idx1)
                    mesh_e2.append(idx2)

                    mesh_e1.append(idx2)
                    mesh_e2.append(idx1)
                
                if j != h-1:
                    idx2 = idx1 + w

                    mesh_e1.append(idx1)
                    mesh_e2.append(idx2)

                    mesh_e1.append(idx2)
                    mesh_e2.append(idx1)

                    if i != 0:
                        idx2 = idx1 + w - 1
                        
                        mesh_e1.append(idx1)
                        mesh_e2.append(idx2)

                        mesh_e1.append(idx2)
                        mesh_e2.append(idx1)

        edges = th.IntTensor([mesh_e1,mesh_e2])
        sf = th.FloatTensor(out)
        i_file = th.FloatTensor(out_i)
        j_file = th.FloatTensor(out_j)
        labels = th.IntTensor(labels)

        th.save(edges, f"{data_dir}/data-{idx}-edges-rect.th")
        th.save(sf, f"{data_dir}/data-{idx}-sf-rect.th")
        th.save(labels, f"{data_dir}/data-{idx}-labels-rect.th")
        outf = open(f"{data_dir}/data-{idx}-meta-rect.txt","w")
        outf.write(f"{w}\n{h}\n")
        outf.close()