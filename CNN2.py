import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from Arch import *
import time
import platform

# Use the MPS backend if available, otherwise fall back to CPU.
if platform.system() == "Darwin":  # macOS
    device =  th.device("cpu") #th.device("mps") if th.backends.mps.is_available() else 
elif platform.system() == "Linux":  # Linux
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
else:  # Default fallback for other systems
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
print("Using device:", device)

def cp_ct_loss(cp_logits, ct_logits, cp_lossfn, ct_lossfn, labelsCP, labelsCT, weightsCP):
    # Coarse loss for all
    loss_cp = cp_lossfn(cp_logits,labelsCP)

    # Fine loss only for samples where true class is B
    is_cp = (labelsCP == 1)
    if is_cp.any():
        loss_ct = ct_lossfn(ct_logits[is_cp],labelsCT[is_cp])
    else:
        loss_ct = th.tensor(0.0)

    return weightsCP[0]*loss_cp + weightsCP[1]*loss_ct

class SFData(Dataset):
    def __init__(self, data_dir, low, high):
        self.data_dir = data_dir
        self.datas = []
        self.freqcp = th.tensor([0,0])
        self.freqct = th.tensor([0,0,0,0])

        for i in range(low,high):
            x = th.load(f"{data_dir}/data-{i}-sf-rect.th")
            x_min = th.min(x)
            x_max = th.max(x)
            x = (x - x_min) / (x_max - x_min)

            yct = th.load(f"{data_dir}/data-{i}-labels-rect.th").to(th.long)
            ycp = th.clone(yct)
            ycp[ycp == 0] = 1
            ycp[ycp == 2] = 1
            ycp[ycp == 3] = 0

            self.freqcp += th.bincount(ycp.reshape(-1,))
            self.freqct += th.bincount(yct.reshape(-1,))

            self.datas.append((x,ycp,yct))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

def measure_fc(model, loader):
    model.eval()
    # good, fp, fn, ft
    
    num_false = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for batch in loader:
        if device.type == 'cuda':
            th.cuda.empty_cache()
        out = model(batch[0].to(device))
        pred = out[0].argmax(dim=1)
        cps = (pred == 1)
        norms = (pred == 0)
        pred[cps] = out[1][cps].argmax(dim=1)
        pred[norms] = 3
        y = batch[2].reshape((-1,))

        for i in range(len(pred)):
            num_false[y[i]][pred[i]] += 1

    titles = ["min","saddle","max","normal"]

    for i in range(4):
        print(f"{titles[i]}\t{num_false[i]}\t{num_false[i][i]/sum(num_false[i])}")
    print(f"overall: {(num_false[0][0] + num_false[1][1] + num_false[2][2] + num_false[3][3]) / (sum(num_false[0]) + sum(num_false[1]) + sum(num_false[2]) + sum(num_false[3]))}")

if __name__ == "__main__":

    num_data = 129
    data_folder = "./real_data"

    train_split = 120
    val_split = 125

    train_graphs = []
    val_graphs = []
    test_graphs = []

    train_data = SFData(data_folder, 0, train_split)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(SFData(data_folder, train_split-5, val_split-5), batch_size=1, shuffle=True)
    test_loader = DataLoader(SFData(data_folder, val_split, num_data), batch_size=1, shuffle=True)

    model = inplaceCNNTwoLevel()
    model = model.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=0.02)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    weightscp = 1.0 / train_data.freqcp
    weightscp = weightscp / sum(weightscp)

    weightsct = 1.0 / train_data.freqct
    weightsct = weightsct / sum(weightsct)
    
    print(weightscp)

    loss_cp = th.nn.CrossEntropyLoss(weight=weightscp.to(device))
    loss_ct = th.nn.CrossEntropyLoss(weight=weightsct.to(device))

    batch_size = 1

    start_time = time.time()
    model.train()
    for epoch in range(100):
        total_loss = 0
        loss = th.FloatTensor([0.0]).to(device)
        batch_idx = 1
        optimizer.zero_grad()        

        for batch in train_loader:
            out_cp, out_ct = model(batch[0].to(device))
            ycp = batch[1].reshape((-1,)).to(device)
            yct = batch[2].reshape((-1,)).to(device)
            loss_ = cp_ct_loss(out_cp, out_ct, loss_cp, loss_ct, ycp, yct, weightscp)
            loss += loss_

            if (batch_idx % batch_size == 0) or (batch_idx == len(train_data)):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss = th.FloatTensor([0.0]).to(device)
            
            batch_idx += 1
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        scheduler.step() # moved here from the epoch. 0.95 -> 0.97
        if device.type == 'cuda':
            th.cuda.empty_cache()
        if epoch % 10 == 0:
            measure_fc(model, val_loader)

    print("test errors:")
    measure_fc(model, test_loader)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    th.save(model.state_dict(), "./model_final.th")
    