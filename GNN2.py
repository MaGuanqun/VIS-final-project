import torch as th
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

from Arch import *

def cp_ct_loss(cp_logits, ct_logits, cp_lossfn, ct_lossfn, labelsCP, labelsCT, weightsCP):
    # coarse_logits: [B, 2]
    # fine_logits:   [B, 3]
    # coarse_labels: [B] — 0 for A, 1 for B
    # fine_labels:   [B] — valid only where coarse == 1

    # Coarse loss for all
    loss_cp = cp_lossfn(cp_logits,labelsCP)

    # Fine loss only for samples where true class is B
    is_cp = (labelsCP == 1)
    if is_cp.any():
        loss_ct = ct_lossfn(ct_logits[is_cp],labelsCT[is_cp])
    else:
        loss_ct = th.tensor(0.0)

    return weightsCP[0]*loss_cp + weightsCP[1]*loss_ct

def measure_fc(model, loader):
    model.eval()
    # good, fp, fn, ft
    
    num_false = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    # critical, noncritical
    # num_false = [[0,0],[0,0]]

    for batch in loader:
        out = model(batch.x, batch.edge_index)
        pred = out[0].argmax(dim=1)
        cps = (pred == 1)
        norms = (pred == 0)
        pred[cps] = out[1][cps].argmax(dim=1)
        pred[norms] = 3

        for i in range(len(pred)):
            num_false[batch.labelsCT[i]][pred[i]] += 1

    titles = ["min","saddle","max","normal"]
    for i in range(4):
        print(f"{titles[i]}\t{num_false[i]}\t{num_false[i][i]/sum(num_false[i])}")
    print(f"overall: {(num_false[0][0] + num_false[1][1] + num_false[2][2] + num_false[3][3]) / (sum(num_false[0]) + sum(num_false[1]) + sum(num_false[2]) + sum(num_false[3]))}")

    # titles = ["cp", "normal"]
    # for i in range(2):
    #     print(f"{titles[i]}\t{num_false[i]}\t{num_false[i][i] / sum(num_false[i])}")
    # print(f"overall: {(num_false[0][0] + num_false[1][1]) / (sum(num_false[0]) + sum(num_false[1]))}")

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

        labelsCP = th.clone(labels)
        labelsCP[labels==0] = 1
        labelsCP[labels==2] = 1
        labelsCP[labels==3] = 0
        # labels = th.stack((labels,labelsCP)).to(th.long)

        # print(sf.shape)
        # print(labels.shape)
        # exit()

        # Create the PyTorch Geometric Data object
        graph = Data(x=sf, edge_index=edges, labelsCP=labelsCP, labelsCT=labels)
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
    model = HybridClassifier()

    optimizer = th.optim.Adam(model.parameters(), lr=0.01)

    # weights = sum(num_each) * 0.25 / num_each
    # weights = weights / sum(weights)

    weightsCP = th.FloatTensor( (1/(num_each[3]), 1/(num_each[0]+num_each[1]+num_each[2]) ))
    weightsCP = weightsCP / sum(weightsCP)

    weightsCT = th.FloatTensor( (1/(num_each[0]), 1/(num_each[1]), 1/(num_each[2]),0 ))
    weightsCT = weightsCT / sum(weightsCT)

    # criterion = th.nn.CrossEntropyLoss(weight=weights)
    cp_loss = th.nn.CrossEntropyLoss(weight=weightsCP)
    ct_loss = th.nn.CrossEntropyLoss(weight=weightsCT)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            cp_out, ct_out = model(batch.x, batch.edge_index)
            # loss = criterion(out, batch.y.to(th.long))
            print(cp_out.shape)
            print(batch.labelsCP.shape)
            loss = cp_ct_loss(cp_out, ct_out, cp_loss, ct_loss, batch.labelsCP, batch.labelsCT, weightsCP)
            print(loss)
            exit()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if epoch % 10 == 0:
            measure_fc(model, val_loader)

    print("test errors:")
    measure_fc(model, test_loader)
    th.save(model.state_dict(), "./model_final.th")