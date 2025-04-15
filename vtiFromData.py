from pyevtk.hl import *
import numpy as np
import torch as th

if __name__ == "__main__":
    data_path = "./data"
    data_idx = 0
    output_name = "./test"

    inf = open(f"{data_path}/data-{data_idx}-meta.txt")
    w = int(inf.readline())
    h = int(inf.readline())

    labels = th.load(f"{data_path}/data-{data_idx}-labels.th")
    sf = th.load(f"{data_path}/data-{data_idx}-sf.th")

    labels = labels.numpy()
    sf = sf.numpy()

    labels = labels.reshape((1,h,w)).T
    sf = sf.reshape((1,h,w)).T

    imageToVTK(output_name, pointData={"sf" : sf, "labels" : labels})