import torch as th
from torch.utils.data import DataLoader
from Arch import inplaceCNNTwoLevel  # Assuming the model architecture is defined in Arch.py
from CNN2 import SFData, measure_fc  # Import SFData and measure_fc from CNN2.py
import platform
import csv

# Use the MPS backend if available, otherwise fall back to CPU.
if platform.system() == "Darwin":  # macOS
    device = th.device("cpu")  # th.device("mps") if th.backends.mps.is_available() else
elif platform.system() == "Linux":  # Linux
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
else:  # Default fallback for other systems
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
print("Using device:", device)




def measure(model, loader):
    model.eval()
    # good, fp, fn, ft
    
    num_false = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for batch in loader:
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
    return pred,y



def save_critical_point(pred,y,shape, path):
    csv_file = []
    csv_file.append('local-minima.csv')
    csv_file.append('saddle.csv')
    csv_file.append('local-maxima.csv')
    
    for cls in range(3):
        correct = [i for i, (predicted, data_item) in enumerate(zip(pred, y)) if predicted == cls and data_item == cls]
        correct_2d = [(i // shape[1], i % shape[1]) for i in correct]
        false_positive = [i for i, (predicted, data_item) in enumerate(zip(pred, y)) if predicted == cls and data_item != cls]
        false_positive_2d = [(i // shape[1], i % shape[1]) for i in false_positive]
        false_negative = [i for i, (predicted, data_item) in enumerate(zip(pred, y)) if predicted != cls and data_item == cls]
        false_negative_2d = [(i // shape[1], i % shape[1]) for i in false_negative]
        
            # Combine all data into a single list with labels
        data_to_save = [(x, y, 1) for x, y in correct_2d] + [(x, y, 2) for x, y in false_positive_2d] +[(x, y, 3) for x, y in false_negative_2d]
        
        with open(path+csv_file[cls], mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "type"])  # Header row
            writer.writerows(data_to_save)

        

# Load the saved model
model = inplaceCNNTwoLevel()
model.load_state_dict(th.load("./model_final.th", map_location=device))
model = model.to(device)
model.eval()

# Prepare the test dataset
data_folder = "./real_data"

name=[]
name.append('/boussinesq')
name.append('/vortex-street')
name.append('/hurricane')
name.append('/CESM')

for i in range(4):    
    test_split_start = i
    test_split_end = i+1
    test_data = SFData(data_folder, test_split_start, test_split_end)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    shape = test_data[0][0].shape
    # Evaluate the model
    print("Testing the model on new data:")
    predicition, ground_truth = measure(model, test_loader)

    save_critical_point(predicition, ground_truth, shape, data_folder+name[i])