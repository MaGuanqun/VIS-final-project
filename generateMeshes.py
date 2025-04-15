import numpy as np
import random
import numpy.random as npr
from pyevtk.hl import *
from math import e
from unionFind import *
import os
import torch as th

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
    w,h,_ = points.shape

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

if __name__ == "__main__":

    data_dir = "./data"

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    num_data = 2000
    offset = 0

    for idx in range(num_data):
        idx += offset
        print(idx)
        w = random.randint(30,100)
        h = random.randint(30,100)

        num_lumps = random.randint(300,800)
        heights = npr.uniform(-0.8,0.8,size=(num_lumps,))
        widths = npr.uniform(100, 1000, size=(num_lumps,))
        x = npr.randint(-10, w+10, size=(num_lumps,))
        y = npr.randint(-10, h+10, size=(num_lumps,))

        out = np.zeros((w,h,1))
        out_i = np.zeros((w,h,1))
        out_j = np.zeros((w,h,1))
        for i in range(w):
            for j in range(h):

                out_i[i,j,0] = i
                out_j[i,j,0] = j

                for k in range(num_lumps):
                    dist = (i-x[k])**2 + (j-y[k])**2
                    if dist < 600:
                        out[i,j,0] += heights[k]*e**(  -dist / widths[k]  )

        labels = np.zeros((w,h,1),dtype=np.int64)
        for i in range(w):
            for j in range(h):
                labels[i,j,0] = classifyPoint(out, (i,j))

        # imageToVTK("./test",pointData={"sf":out})

        min_out = np.min(out)
        max_out = np.max(out)
        out = (out - min_out) / (max_out - min_out)

        out = out.T.reshape(-1)
        out_i = out_i.T.reshape(-1)
        out_j = out_j.T.reshape(-1)

        labels = labels.T.reshape(-1)

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

        th.save(edges, f"{data_dir}/data-{idx}-edges.th")
        th.save(sf, f"{data_dir}/data-{idx}-sf.th")
        th.save(i_file, f"{data_dir}/data-{idx}-i.th")
        th.save(j_file, f"{data_dir}/data-{idx}-j.th")
        th.save(labels, f"{data_dir}/data-{idx}-labels.th")

        outf = open(f"{data_dir}/data-{idx}-meta.txt","w")
        outf.write(f"{w}\n{h}\n")
        outf.close()