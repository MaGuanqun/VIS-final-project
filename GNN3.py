import torch as th
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

from Arch import *

def complex_loss(min_logits, max_logits, reg_logits, min_lossfn, max_lossfn, other_lossfn, labels_mins, labels_maxs, labels_other, labels_all, weights):
    not_min = (labels_mins == 0)
    not_max = (labels_maxs[not_min] == 0)

    loss_min = min_lossfn(min_logits,labels_mins)

    if not_min.any():
        loss_max = max_lossfn(max_logits[not_min],labels_maxs[not_min])

        if not_max.any():
            loss_other = other_lossfn(reg_logits[not_min][not_max],labels_other[not_min][not_max])
        else:
            loss_max = th.tensor(0.0)

    else:
        loss_other = th.tensor(0.0)
        loss_max = th.tensor(0.0)


    return weights[0]*loss_min + weights[2]*loss_max + (weights[1]+weights[3])*loss_other

def measure_fc(model, loader):
    model.eval()
    # good, fp, fn, ft
    
    num_false = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    # critical, noncritical
    # num_false = [[0,0],[0,0]]

    for batch in loader:
        min_l, max_l, other_l = model(batch.x, batch.edge_index)
        pred_min = min_l.argmax(dim=1)
        pred_max = max_l.argmax(dim=1)
        pred_other = other_l.argmax(dim=1)

        for i in range(len(pred_min)):
            if pred_min[i] == 1:
                pred = 0
            elif pred_max[i] == 1:
                pred = 2
            elif pred_other[i] == 1:
                pred = 3
            else:
                pred = 1

            num_false[batch.y[i]][pred] += 1

    titles = ["min","saddle","max","normal"]
    for i in range(4):
        print(f"{titles[i]}\t{num_false[i]}\t{num_false[i][i]/sum(num_false[i])}")
    print(f"overall: {(num_false[0][0] + num_false[1][1] + num_false[2][2] + num_false[3][3]) / (sum(num_false[0]) + sum(num_false[1]) + sum(num_false[2]) + sum(num_false[3]))}")

if __name__ == "__main__":

    num_data = 2000
    data_folder = "./data"

    train_split = 1799
    val_split = 1899

    train_graphs = []
    val_graphs = []
    test_graphs = []

    num_each = th.Tensor([0,0,0,0]).to(th.int64)

    for i in range(num_data):

        sf = th.load(f"{data_folder}/data-{i}-sf.th")

        edges = th.load(f"{data_folder}/data-{i}-edges.th").to(th.int64)
        labels = th.load(f"{data_folder}/data-{i}-labels.th").to(th.long)
        num_each += th.bincount(labels)

        labels_min = th.clone(labels)
        labels_min[labels==0] = 1
        labels_min[labels==1] = 0
        labels_min[labels==2] = 0
        labels_min[labels==3] = 0

        labels_max = th.clone(labels)
        labels_max[labels==0] = 0
        labels_max[labels==1] = 0
        labels_max[labels==2] = 1
        labels_max[labels==3] = 0

        labels_other = th.clone(labels)
        labels_other[labels==0] = 0
        labels_other[labels==1] = 0
        labels_other[labels==2] = 0
        labels_other[labels==3] = 1

        # Create the PyTorch Geometric Data object
        graph = Data(x=sf, edge_index=edges, y=labels, mins=labels_min, maxs=labels_max, others=labels_other)
        if i <= train_split:
            train_graphs.append(graph)
        elif i <= val_split:
            val_graphs.append(graph)
        else:
            test_graphs.append(graph)

    train_loader = DataLoader(train_graphs, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=100, shuffle=True)

    # model = GraphNodeClassifier(1,4)
    model = HybridClassifierLarge()

    optimizer = th.optim.Adam(model.parameters(), lr=0.01)

    weights = sum(num_each) * 0.25 / num_each
    weights = weights / sum(weights)

    weights_min = th.FloatTensor((1/(num_each[1]+num_each[2]+num_each[3]), 1/num_each[0]))
    weights_max = th.FloatTensor((1/(num_each[0]+num_each[1]+num_each[3]), 1/num_each[2]))
    weights_other = th.FloatTensor((1/(num_each[1]), 1/(num_each[3])))

    min_loss = th.nn.CrossEntropyLoss(weight=weights_min)
    max_loss = th.nn.CrossEntropyLoss(weight=weights_max)
    other_loss = th.nn.CrossEntropyLoss(weight=weights_other)

    model.train()
    measure_fc(model, test_loader)
    for epoch in range(100):
        total_loss = 0     
        for batch in train_loader:
            optimizer.zero_grad()
            min_l, max_l, other_l = model(batch.x, batch.edge_index)
            loss = complex_loss(min_l, max_l, other_l, min_loss, max_loss, other_loss, batch.mins, batch.maxs, batch.others, batch.y, weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if epoch % 10 == 0:
            measure_fc(model, val_loader)

    print("test errors:")
    measure_fc(model, test_loader)
    th.save(model.state_dict(), "./model_final.th")