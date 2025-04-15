from pyevtk.hl import *
import numpy as np
import torch as th
from Arch import *

if __name__ == "__main__":
    data_path = "./data"
    data_idx = 1951
    output_name = "./test"
    model_class = inplaceCNNTwoLevel
    model_params = "./normalized-98.th"

    sf_rect = th.load(f"{data_path}/data-{data_idx}-sf-rect.th")
    sf_rect_min = th.min(sf_rect)
    sf_rect_max = th.max(sf_rect)
    sf_rect = (sf_rect - sf_rect_min) / (sf_rect_max - sf_rect_min)

    labels_rect = th.load(f"{data_path}/data-{data_idx}-labels-rect.th")

    h,w = sf_rect.shape

    model = model_class()
    model.load_state_dict(th.load(model_params, weights_only=True))
    model.eval()
    labels_mine = model(sf_rect.reshape((1,h,w)))

    pred = labels_mine[0].argmax(dim=1)
    cps = (pred == 1)
    norms = (pred == 0)
    pred[cps] = labels_mine[1][cps].argmax(dim=1)
    pred[norms] = 3
    y = labels_rect.reshape((-1,))

    labels_rect = labels_rect.reshape((-1,))

    good = 0
    bad = 0
    for i in range(len(pred)):
        if pred[i] == labels_rect[i]:
            good += 1
        else:
            bad += 1
    
    print((good,bad))

    sf = sf_rect.numpy().reshape((1,h,w)).T
    labels = labels_rect.numpy().reshape((1,h,w)).T
    pred = pred.numpy().reshape((1,h,w)).T

    imageToVTK(output_name, pointData={"sf" : sf, "labels" : labels, "labelsCT" : pred})