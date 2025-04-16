from pyevtk.hl import *
import numpy as np
import torch as th

if __name__ == "__main__":
    data_path = "./real_data"
    data_idx = 55
    output_name = "./test/test"

    inf = open(f"{data_path}/data-{data_idx}-meta-rect.txt")
    w = int(inf.readline())
    h = int(inf.readline())

    labels = th.load(f"{data_path}/data-{data_idx}-labels-rect.th")
    sf = th.load(f"{data_path}/data-{data_idx}-sf-rect.th")
    critical_points = th.load(f"{data_path}/data-{data_idx}-edges-rect.th")

    labels = labels.numpy()
    sf = sf.numpy()
    
    labels = np.expand_dims(labels, axis=2)
    sf = np.expand_dims(sf, axis=2)

    # labels = labels.reshape((1,h,w)).T
    # sf = sf.reshape((1,h,w)).T

    imageToVTK(output_name, pointData={"sf" : sf, "labels" : labels})