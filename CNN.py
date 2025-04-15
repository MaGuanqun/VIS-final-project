import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from Arch import *

class SFData(Dataset):
    def __init__(self, data_dir, low, high):
        self.data_dir = data_dir
        self.datas = []
        self.num_each = th.tensor([0,0,0,0])

        for i in range(low,high):
            x = th.load(f"{data_dir}/data-{i}-sf-rect.th")

            x_min = th.min(x)
            x_max = th.max(x)
            x = (x - x_min) / (x_max - x_min)

            y = th.load(f"{data_dir}/data-{i}-labels-rect.th").to(th.long)
            self.num_each += th.bincount(y.reshape(-1))
            self.datas.append((x,y))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

def measure_fc(model, loader):
    model.eval()
    # good, fp, fn, ft
    
    num_false = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for batch in loader:
        out = model(batch[0])
        pred = out.argmax(dim=1)
        pred = pred.reshape((-1,))
        y = batch[1].reshape((-1,))

        for i in range(len(pred)):
            num_false[y[i]][pred[i]] += 1

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

    train_split = 1800
    val_split = 1900

    train_graphs = []
    val_graphs = []
    test_graphs = []

    train_data = SFData(data_folder, 0, train_split)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(SFData(data_folder, train_split, val_split), batch_size=1, shuffle=True)
    test_loader = DataLoader(SFData(data_folder, val_split, num_data), batch_size=1, shuffle=True)

    model = inplaceCNN2()
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.97) #98 for moderately big, 97 for very big.

    weights = 1.0 / train_data.num_each
    weights = weights / sum(weights)

    criterion = th.nn.CrossEntropyLoss(weight=weights)

    batch_size = 50

    model.train()
    for epoch in range(100):
        total_loss = 0
        loss = th.FloatTensor([0.0])
        batch_idx = 1
        optimizer.zero_grad()        
        for batch in train_loader:
            out = model(batch[0])
            y = batch[1].reshape((-1,))
            loss_ = criterion(out, y)
            loss += loss_

            if (batch_idx % batch_size == 0) or (batch_idx == len(train_data)):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss = th.FloatTensor([0.0])
            
            batch_idx += 1
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        scheduler.step() # moved here from the epoch. 0.95 -> 0.97
        if epoch % 10 == 0:
            measure_fc(model, val_loader)

    print("test errors:")
    measure_fc(model, test_loader)
    th.save(model.state_dict(), "./model_final.th")