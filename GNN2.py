import torch as th
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

from Arch import GraphNodeClassifier

def measure_fc(model, loader):
    model.eval()
    # good, fp, fn, ft
    freq = [0,0,0,0]
    num_false = [0,0,0,0]

    # critical, noncritical
    # freq = [0,0]
    # good, fp, fn
    # num_false = [0,0,0]

    for batch in test_loader:
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        
        for i in range(len(out)):
            freq[pred[i]] += 1
            if pred[i] == batch.y[i]:
                num_false[0] += 1
            elif batch.y[i] == 3:
                num_false[1] += 1
            elif pred[i] == 3:
                num_false[2] += 1
            else:
                num_false[3] += 1
    
        # for i in range(len(out)):
        #     freq[pred[i]] += 1
        #     if pred[i] == batch.y[i]:
        #         num_false[0] += 1
        #     elif batch.y[i] == 0:
        #         num_false[1] += 1
        #     else:
        #         num_false[2] += 1
    model.train()
    acc = num_false[0] / sum(num_false)
    return freq,num_false,acc

if __name__ == "__main__":

    num_data = 1000
    data_folder = "./data"

    train_split = 799
    val_split = 899

    weight_op = 0.1
    weight_cp = (1.0 - weight_op)/3

    train_graphs = []
    val_graphs = []
    test_graphs = []

    num_each = th.Tensor([0,0,0,0]).to(th.int64)

    for i in range(num_data):

        sf = th.load(f"{data_folder}/data-{i}-sf.th")
        edges = th.load(f"{data_folder}/data-{i}-edges.th").to(th.int64)
        labels = th.load(f"{data_folder}/data-{i}-labels.th")
        # labels[labels==1] = 1
        # labels[labels==2] = 0
        # labels[labels==3] = 1
        num_each += th.bincount(labels)

        # Create the PyTorch Geometric Data object
        graph = Data(x=sf, edge_index=edges, y=labels)
        if i <= train_split:
            train_graphs.append(graph)
        elif i <= val_split:
            val_graphs.append(graph)
        else:
            test_graphs.append(graph)

    train_loader = DataLoader(train_graphs, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=100, shuffle=True)

    model = GraphNodeClassifier(in_channels=1, out_channels=4)

    optimizer = th.optim.Adam(model.parameters(), lr=0.01)

    weights = sum(num_each) * 0.25 / num_each
    weights = weights / sum(weights)

    criterion = th.nn.CrossEntropyLoss(weight=weights)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.to(th.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if epoch % 10 == 0:
            print(measure_fc(model, val_loader))

    print("test errors:")
    print(measure_fc(model, test_loader))
    # th.save(model.state_dict(), "./model_final.th")