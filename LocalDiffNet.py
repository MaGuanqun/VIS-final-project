import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import csv

import platform
import time
import random

# Set random seeds for reproducibility.
# np.random.seed(42)
# torch.manual_seed(42)

## fix the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





# Use the MPS backend if available, otherwise fall back to CPU.
if platform.system() == "Darwin":  # macOS
    device =  torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") 
    # num_cores = os.cpu_count()
    # print("Using", num_cores, "CPU cores.")
    # Set PyTorch to use all available cores for intra-operator parallelism.
    # torch.set_num_threads(num_cores)
# And optionally for inter-operator parallelism.
    # torch.set_num_interop_threads(num_cores)
elif platform.system() == "Linux":  # Linux
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:  # Default fallback for other systems
    device = torch.device("cpu")
    
print("Using device:", device)
# # ---------------------------


class Data:
    def __init__(self, x, edge_index, y):
        self.x = x         
        self.edge_index = edge_index  
        self.y = y          


def load_mesh_data(mesh_id, data_dir="."):
    """
    Loads the mesh data from pre-saved .th files.
    Expects the following files in data_dir:
      - data-{mesh_id}-edges.th : Edge index tensor (shape [2, num_edges])
      - data-{mesh_id}-sf.th    : Scalar field values, one value per node.
      - data-{mesh_id}-labels.th: Labels for each node,
                                 using: 0 (local minima), 1 (saddle), 2 (local maxima), 3 (general)
    """
    edges_file = os.path.join(data_dir, f"data-{mesh_id}-edges.th")
    sf_file    = os.path.join(data_dir, f"data-{mesh_id}-sf.th")
    labels_file= os.path.join(data_dir, f"data-{mesh_id}-labels.th")
    size_file = os.path.join(data_dir, f"data-{mesh_id}-meta.txt")

    # Read the mesh size from the meta file.
    with open(size_file, "r") as f:
        lines = f.readlines()
        H = int(lines[0].strip())
        W = int(lines[1].strip())
        size = [H, W]
    # Read the mesh data from the files.
    

    edge_index = torch.load(edges_file, weights_only=True).to(torch.int64)

    sf = torch.load(sf_file, weights_only=True)
    sf = (sf - sf.min()) / (sf.max() - sf.min())  # Normalize values to the range [0, 1]

    labels = torch.load(labels_file, weights_only=True)
    labels[labels == -1] = 3

    # Ensure the scalar field is formatted as a 2D tensor [num_nodes, 1]
    if sf.dim() == 1:
        x = sf.view(-1, 1)
    else:
        x = sf

    return Data(x=x, edge_index=edge_index, y=labels),size


class MeshLocalDataset(Dataset):
    """
    Dataset to extract local patches for each node.
    For each node in the mesh, we retrieve its scalar value (center)
    and the 6 neighbors in the fixed order:
      left, right, top, bottom, top right, and bottom left.
    Nodes on boundaries missing a neighbor are padded with the center value.
    """
    def __init__(self, mesh_id, data_dir="."):
        # Load full mesh data and grid size.
        self.data, self.size = load_mesh_data(mesh_id, data_dir)
        self.x = self.data.x  # shape: [num_nodes, 1]
        self.y = self.data.y  # shape: [num_nodes]
        self.H, self.W = self.size
        self.num_nodes = self.x.shape[0]

    def __len__(self):
        return self.num_nodes

    def index_to_coord(self, idx):
        """Given a node index, compute its (row, col) assuming row-major order."""
        row = idx // self.W
        col = idx % self.W
        return row, col

    def coord_to_index(self, row, col):
        """Convert (row, col) to the linear index."""
        return row * self.W + col

    def get_neighbor_value(self, row, col, neighbor_offset, center_value):
        """
        Given a coordinate (row, col) and a neighbor offset (d_row, d_col),
        return the scalar value of the neighbor if inside the grid; otherwise return center_value.
        """
        n_row, n_col = row + neighbor_offset[0], col + neighbor_offset[1]
        if 0 <= n_row < self.H and 0 <= n_col < self.W:
            idx = self.coord_to_index(n_row, n_col)
            return self.x[idx, 0].item()
        else:
            # Fallback: if neighbor is out of bounds, use the center value.
            return center_value

    def __getitem__(self, idx):
        """
        For node at index idx, return a tuple of:
         (center_value_tensor, neighbors_tensor, label)
        neighbors_tensor is ordered as:
         [left, right, top, bottom, top right, bottom left]
        """
        center_value = self.x[idx, 0].item()
        label = int(self.y[idx].item())
        row, col = self.index_to_coord(idx)

        # Define neighbor offsets corresponding to:
        # left, right, top, bottom, top right, bottom left.
        # (row offset, column offset)
        neighbor_offsets = [
            (0, -1),   # left
            (1, -1),    # bottom left
            (1, 0),    # bottom
            (0, 1),    # right
            (-1, 1),   # top right
            (-1, 0)   # top
        ]
        differences = []
        for offset in neighbor_offsets:
            n_val = self.get_neighbor_value(row, col, offset, center_value)
            differences.append(center_value-n_val)
        diffs_tensor = torch.tensor(differences, dtype=torch.float32)
        return diffs_tensor, label

class EnhancedDiscreteMorseCriticalPointNet(nn.Module):
    def __init__(self,
                 input_dim=6,
                 raw_hidden=16,
                 conv_channels=8,
                 combined_hidden=32,
                 num_classes=4,
                 dropout_prob=0.2):
        """
        This network uses:
          - A raw branch (FC layers) processing the 6 differences.
          - A multi-scale convolution branch with two parallel 1D conv paths:
              * One with kernel size 3, dilation=1.
              * One with kernel size 3, dilation=2.
            Both branches use cyclic padding so that patterns wrapping around
            the 6-element sequence are captured.
          - The outputs of both branches are mapped to a common dimension and concatenated.
          - Finally, an MLP (with dropout) produces the class logits.
        """
        super(EnhancedDiscreteMorseCriticalPointNet, self).__init__()
        # Raw branch.
        self.raw_branch = nn.Sequential(
            nn.Linear(input_dim, raw_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # Convolution branch:
        # For a kernel size of 3, we manually apply cyclic padding.
        # Branch 1: dilation=1.
        self.conv_branch1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=6, dilation=1, padding=0)
        # Branch 2: dilation=2.
        self.conv_branch2 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=6, dilation=2, padding=0)
        # Fully connected layer to map concatenated conv features to dimension raw_hidden.
        self.fc_conv = nn.Linear(conv_channels * 2, raw_hidden)
        # Final classifier: combine raw and conv features.
        self.combined_fc = nn.Sequential(
            nn.Linear(raw_hidden * 2, combined_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(combined_hidden, num_classes)
        )
    
    def cyclic_pad(self, x, pad_width):
        """
        x: [B, C, L]
        pad_width: int, number of elements to take from the end (for left side padding).
        Returns x padded cyclically on the left.
        """
        pad = x[:, :, -pad_width:]
        return torch.cat([pad, x], dim=2)
    
    def forward(self, x):
        # x: [B, 6]
        B = x.shape[0]
        # Raw branch.
        raw_features = self.raw_branch(x)  # [B, raw_hidden]
        
        # Convolution branch requires input of shape [B, 1, 6].
        x_unsqueeze = x.unsqueeze(1)  # [B, 1, 6]
        # Process branch 1: dilation=1, kernel size=3 -> need to cyclically pad 2 elements on the left.
        x1 = self.cyclic_pad(x_unsqueeze, pad_width=2)  # [B, 1, 8]
        conv1 = self.conv_branch1(x1)  # output length: 8 - 3 + 1 = 6
        conv1 = F.relu(conv1)  # [B, conv_channels, 6]
        
        # Process branch 2: dilation=2, kernel size=3, effective receptive field 5.
        # For dilation=2, we need to pad (kernel_size-1)*dilation = 4 on the left.
        x2 = self.cyclic_pad(x_unsqueeze, pad_width=4)  # [B, 1, 10]
        conv2 = self.conv_branch2(x2)  # output length: 10 - (3-1)*2 - 3 + 1 = 6 (check: 10 - 4 - 3 + 1 = 4; adjust manually)
        # Note: For dilation=2 and kernel size=3, the output length is computed as:
        # L_out = L_in + 2*padding - dilation*(kernel_size-1) - 1 + 1.
        # Here, with effective L_in=10 and no built-in padding, we expect L_out = 10 - 4 = 6.
        conv2 = F.relu(conv2)  # [B, conv_channels, 6]
        
        # Global average pooling along sequence dimension for both conv outputs.
        conv1_avg = torch.mean(conv1, dim=2)  # [B, conv_channels]
        conv2_avg = torch.mean(conv2, dim=2)    # [B, conv_channels]
        conv_concat = torch.cat([conv1_avg, conv2_avg], dim=1)  # [B, conv_channels*2]
        conv_features = self.fc_conv(conv_concat)  # [B, raw_hidden]
        
        # Fuse raw and conv branch features.
        combined = torch.cat([raw_features, conv_features], dim=1)  # [B, raw_hidden*2]
        logits = self.combined_fc(combined)  # [B, num_classes]
        return logits

class DiscreteMorseCriticalPointNet(nn.Module):
    def __init__(self,
                 input_dim=6,
                 raw_hidden=8,
                 conv_channels=8,
                 combined_hidden=16,
                 num_classes=4):
        """
        This network processes a 6-dimensional difference vector
        using two branches:
          (a) The "raw branch" processes the 6 differences directly via a fully connected layer.
          (b) The "conv branch" applies a 1D convolution over the cyclic sequence of differences.
              Cyclic padding is performed so that local patterns across the boundary of the sequence
              (i.e. considering d5 next to d0) are captured.
        These branches are concatenated and fed to a final MLP.
        The convolution branch is intended to help the network recognize subtle saddle patterns
        (e.g. [+,-,-,+,-,-], [–,+,+,–,+,+], [–,+,+,+,–,+]) encountered in discrete Morse theory.
        """
        super(DiscreteMorseCriticalPointNet, self).__init__()
        # Raw branch: simple MLP from 6 inputs.
        self.raw_branch = nn.Sequential(
            nn.Linear(input_dim, raw_hidden),
            nn.ReLU()
        )
        # Convolution branch:
        # We'll treat the 6 differences as a 1-channel sequence of length 6.
        # To implement cyclic padding, we will manually pad the input.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=6, padding=0)
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=0)
        # A fully connected layer to map conv features to the same dimension as raw branch output.
        self.fc_conv = nn.Linear(conv_channels, raw_hidden)
        
        # Final classifier: combine raw and conv branch features.
        self.combined_fc = nn.Sequential(
            nn.Linear(raw_hidden * 2, combined_hidden),
            nn.ReLU(),
            nn.Linear(combined_hidden, num_classes)
        )
        
    def forward(self, x):
        # x: [batch_size, 6]
        # Raw branch.
        raw_features = self.raw_branch(x)  # shape [B, raw_hidden]

        # Convolution branch.
        # Reshape input for convolution: [B, 1, 6]
        x_conv = x.unsqueeze(1)
        # Perform cyclic padding manually.
        # For kernel_size 3, we need to pad 2 values from the end at the left.
        pad = x_conv[:, :, -2:]  # shape [B, 1, 2]
        x_conv = torch.cat([pad, x_conv], dim=2)  # shape becomes [B, 1, 8]
        # First convolution layer.
        x_conv = self.conv1(x_conv)  # output length: 8 - 3 + 1 = 6
        x_conv = F.relu(x_conv)
        # Again cyclically pad before the second convolution.
        pad2 = x_conv[:, :, -2:]
        x_conv = torch.cat([pad2, x_conv], dim=2)  # [B, conv_channels, 6+2=8]
        x_conv = self.conv2(x_conv)  # output length: 8 - 3 + 1 = 6
        x_conv = F.relu(x_conv)
        # Global average pool over the sequence dimension.
        conv_out = torch.mean(x_conv, dim=2)  # shape [B, conv_channels]
        conv_features = self.fc_conv(conv_out)  # shape [B, raw_hidden]

        # Combine raw and convolutional features.
        combined = torch.cat([raw_features, conv_features], dim=1)  # shape [B, raw_hidden*2]
        logits = self.combined_fc(combined)  # shape [B, num_classes]
        return logits

class LocalDiffNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=4):
        """
        The network uses the central value and the 6 differences
        (neighbor minus center) as features.
        """
        super(LocalDiffNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], num_classes)
    
    def forward(self, diffs):
        # diffs: [batch_size, 6]
        x = torch.tanh(self.fc1(diffs))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits  # raw scores



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for diffs, labels in dataloader:
        diffs = diffs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(diffs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * diffs.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    num_classes = 4  
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    accuracy = [0] * num_classes
    with torch.no_grad():
        for diffs, labels in dataloader:
            diffs = diffs.to(device)
            labels = labels.to(device)
            logits = model(diffs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * diffs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for cls in range(num_classes):
                correct_per_class[cls] += ((predicted == cls) & (labels == cls)).sum().item()
                total_per_class[cls] += (labels == cls).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    for cls in range(num_classes):
        accuracy[cls] = correct_per_class[cls] / total_per_class[cls] if total_per_class[cls] > 0 else 0
    return epoch_loss, epoch_acc, accuracy



def save_data(data_item, idx,test_data_size,predicted):
     # (A) Export the test scalar field as a CSV file.
    field = data_item.x.cpu().numpy()  
    field_csv_filename = f"test_fields_csv/test_field_{idx}.csv"
    np.savetxt(field_csv_filename, field, delimiter=",")
    print(f"Test scalar field saved to {field_csv_filename}")
    
    # (B) Extract predicted critical points.
    critical_points = []
    W = test_data_size[idx][0]
    H = test_data_size[idx][1]
    for i in range(predicted.shape[0]):
        val = predicted[i].item() if predicted[i].ndim == 0 else predicted[i].squeeze().item()
        if val < 3:
            row = i // W
            col = i % W
            critical_points.append((row, col, field[i], val))
    points_csv_filename = f"extracted_points/extracted_points_{idx}.csv"
    with open(points_csv_filename, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["x", "y", "scalar", "type"])
        csvwriter.writerows(critical_points)
    print(f"Extracted critical points saved to {points_csv_filename}")
    
    # (C) Export the test scalar field as an OBJ mesh using Paraview triangulation.
    # Triangulation: For each cell (with vertices: top-left (v1), top-right (v2), bottom-right (v3), bottom-left (v4)):
    #   - Triangle 1: (v1, v2, v4)
    #   - Triangle 2: (v2, v3, v4)
    obj_filename = f"test_fields_obj/test_field_{idx}.obj"
    with open(obj_filename, mode="w") as f_obj:
        # Write vertices.
        for r in range(H):
            for c in range(W):
                x_coord = r      # row index
                y_coord = c      # column index
                z_coord = field[r*W+c]  # scalar value as z
                f_obj.write(f"v {x_coord} {y_coord} {z_coord}\n")
        # Write faces.
        for r in range(H - 1):
            for c in range(W - 1):
                v1 = r * W + c + 1        
                v2 = r * W + (c + 1) + 1     
                v3 = (r + 1) * W + (c + 1) + 1 
                v4 = (r + 1) * W + c + 1     
                # Triangle 2: top-right, bottom-right, bottom-left.
              
                f_obj.write(f"f {v1} {v2} {v3}\n")
                # Triangle 1: top-left, top-right, bottom-left.
                f_obj.write(f"f {v1} {v3} {v4}\n")
    print(f"Test mesh OBJ saved to {obj_filename}")





num_train = 5  # Number of training scalar fields.
num_test = 5   # Number of test scalar fields.
batch_size = 5  # Batch size for training.
data_dir = "data"  # Directory containing the mesh data files.
train_data_list = []
test_data_list = []
train_data_size=[]
test_data_size=[]

# Generate training fields.
for idx in range(num_train):
    dataset = MeshLocalDataset(idx, data_dir=data_dir)
    train_data_list.append(dataset)
    train_data_size.append(dataset.size)

# Generate test fields.
for idx in range(num_test):
    dataset = MeshLocalDataset(idx+num_train, data_dir=data_dir)
    test_data_list.append(dataset)
    test_data_size.append(dataset.size)

train_dataset = ConcatDataset(train_data_list)
test_dataset = ConcatDataset(test_data_list)

# Create DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Compute the positive weight from the training data to mitigate class imbalance.
num_classes = 4  
class_counts = [0] * num_classes
for data in train_data_list:
    for lbl in data.y.view(-1).tolist():
        class_counts[lbl] += 1

total_labels = sum(class_counts)
# Compute weights so that a class's weight is inversely proportional to its frequency.
class_weights = [total_labels / (count + 1e-8) for count in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print(f"Total training nodes: {sum(data.y.numel() for data in train_data_list)}")
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights_tensor}")



hidden_channels = [16, 8,32]  # Hidden channels for the GCN layers.

model = DiscreteMorseCriticalPointNet().to(device)
# model = LocalDiffNet(input_dim=6,hidden_dim=hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))#


num_epochs = 100        # Measure training time
start_time = time.time()


for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        test_loss, test_acc, class_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"  Test  Acc: {test_acc:.4f}, Class Accuracies: {', '.join(f'{acc:.4f}' for acc in class_accuracy)}")


end_time = time.time()
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

# Save the trained model to a file.
model_save_path = data_dir+"/local_diff_net.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ---------------------------
# 6. Test Export (for Reference)
# ---------------------------
model.eval()
os.makedirs("test_fields_csv", exist_ok=True)    # CSV files for scalar fields (test data).
os.makedirs("extracted_points", exist_ok=True)     # CSV files for extracted critical points (test data).
os.makedirs("test_fields_obj", exist_ok=True)      # OBJ mesh files (test data).


test_loss, test_acc, class_accuracy = evaluate(model, test_loader, criterion, device)

print("Test Accuracy: {:.4f}".format(test_acc))

# Calculate and print accuracy for each class
for cls in range(num_classes):
    print(f"Class {cls} Accuracy: {class_accuracy[cls]:.4f}")

