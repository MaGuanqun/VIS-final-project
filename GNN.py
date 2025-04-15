import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,GraphConv,GATConv
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
    device =  torch.device("cpu") #torch.device("mps") if torch.backends.mps.is_available() else
    num_cores = os.cpu_count()
    print("Using", num_cores, "CPU cores.")
    # Set PyTorch to use all available cores for intra-operator parallelism.
    torch.set_num_threads(num_cores)
# And optionally for inter-operator parallelism.
    torch.set_num_interop_threads(num_cores)
elif platform.system() == "Linux":  # Linux
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:  # Default fallback for other systems
    device = torch.device("cpu")
    
print("Using device:", device)
# # ---------------------------

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





# # ---------------------------
# # 2. Utility: Create a PyG Data Object from a Scalar Field Using Triangulation Neighbors
# # ---------------------------
# def create_field_data(field):
#     """
#     Given a numpy scalar field of shape (H, W), this function returns a PyTorch Geometric
#     Data object that uses the triangulation connectivity.
#       - x: node features (the scalar value at each grid point, as a column vector)
#       - y: binary labels for each node. A node is labeled as critical (1) if its scalar value is strictly 
#            greater than the scalar values at all neighbors determined by the triangulation connectivity.
#       - edge_index: the triangulation edge index computed above.
#     """
#     # Node features.
#     x = torch.tensor(field.reshape(-1, 1), dtype=torch.float)
#     y_list = []
#     num_vertices = H * W
#     for i in range(num_vertices):
#         current_val = field[i // H, i % H]
#         # Get neighbors from the neighbor dictionary; if no neighbor found, use empty list.
#         neighbors = neighbor_dict.get(i, [])

#         # print(f"Node {i}: current_val={current_val}, neighbors={neighbors}")
#         if neighbors:
#             neighbor_vals = [field[n // H, n % H] for n in neighbors]
#             # print(current_val, neighbor_vals)
            
#             is_critical = 1 if current_val > max(neighbor_vals) else 0
#         else:
#             is_critical = 0
#         y_list.append(is_critical)
#     y = torch.tensor(y_list, dtype=torch.float)
#     return Data(x=x, edge_index=tri_edge_index, y=y)



class GraphNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_p=0.3, use_residual=False):
        super(GraphNN, self).__init__()
        # Use attention-based graph convolutions.
        self.conv1 = GATConv(in_channels, hidden_channels[0])
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1])
        # self.conv3 = GATConv(hidden_channels[1], hidden_channels[2])
        self.conv4 = GATConv(hidden_channels[1], 4)  # 4 output classes
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First attention layer.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.dropout(x)
        # Second attention layer.
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = self.dropout(x)

        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.dropout(x)
        # Final layer to get logits.
        x = self.conv4(x, edge_index)
        return x


# # GraphConv-based Node Classifier
# class GraphNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout_p=0.3, use_residual=False):
#         super(GraphNN, self).__init__()

#         self.conv1 = GraphConv(in_channels, hidden_channels[0])
#         self.Linear1 = nn.Linear(hidden_channels[0],hidden_channels[1])
#         self.conv2 = GraphConv(hidden_channels[1],hidden_channels[2])
#         self.Linear2 = nn.Linear(hidden_channels[2],4)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = x.reshape((-1,1))
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.Linear1(x)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.Linear2(x)
#         return x


# class GraphNN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, dropout_p=0.3, use_residual=False):
#         """
#         Modified network structure with:
#           - Three GCNConv layers to increase depth.
#           - Batch Normalization after the first two layers.
#           - Dropout for regularization.
#           - Optionally adding a residual connection from the input.
          
#         Parameters:
#           - in_channels: Number of input features (1 in this case).
#           - hidden_channels: List of hidden units for each layer (e.g., [32, 16]).
#           - dropout_p: Dropout probability.
#           - use_residual: If True, adds a skip connection from the input features to the output.
#         """
#         super(GraphNN, self).__init__()
#         self.use_residual = use_residual

#         # First Graph Convolution layer.
#         self.conv1 = GCNConv(in_channels, hidden_channels[0])
#         self.bn1 = nn.BatchNorm1d(hidden_channels[0])

#         # Second layer increases the model depth.
#         self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
#         self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        
#         # Output layer with 4 classes.
#         self.conv4 = GCNConv(hidden_channels[1], 4)
        
#         # Dropout parameter to help regularize the network.
#         self.dropout = nn.Dropout(p=dropout_p)

#         # Optional residual connection: project input features to match output size.
#         if use_residual:
#             self.res_fc = nn.Linear(in_channels, 4)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         h = F.relu(self.bn1(self.conv1(x, edge_index)))
#         h = self.dropout(h)
#         h = F.relu(self.bn2(self.conv2(h, edge_index)))
#         h = self.dropout(h)
#         h = self.conv4(h, edge_index)
#         if self.use_residual:
#             # Directly projects x to one logit per node and adds it.
#             h = h + self.res_fc(x)
#         return h  # Shape: [num_nodes, 4]



num_train = 500  # Number of training scalar fields.
num_test = 10   # Number of test scalar fields.
data_dir = "data"  # Directory containing the mesh data files.
train_data_list = []
test_data_list = []
train_data_size=[]
test_data_size=[]

# Generate training fields.
for idx in range(num_train):
    data_obj,size = load_mesh_data(idx, data_dir)
    train_data_list.append(data_obj)
    train_data_size.append(size)

# Generate test fields.
for idx in range(num_test):
    data_obj,size = load_mesh_data(num_train + idx, data_dir)
    test_data_list.append(data_obj)
    test_data_size.append(size)


#############################################
# 3. Compute Class Weights (Optional)
#############################################
# There are 4 classes: 0 (minima), 1 (saddle), 2 (maxima), 3 (general).
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

# # Compute the positive weight from the training data to mitigate class imbalance.
# total_pos, total_count = 0, 0
# for data in train_data_list:
#     total_pos += data.y.sum().item()
#     total_count += data.y.shape[0]
# total_neg = total_count - total_pos
# pos_weight_value = total_neg / (total_pos + 1e-8)  # Avoid division by zero.
# pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)
# print(f"Training samples: {total_count}, Positive: {total_pos}, Negative: {total_neg}, pos_weight: {pos_weight.item():.2f}")

# Create DataLoaders.
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
# We process test (and later training export) one graph at a time.
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)


hidden_channels = [16, 32,32]  # Hidden channels for the GCN layers.

model = GraphNN(in_channels=1, hidden_channels=hidden_channels, dropout_p=0.3, use_residual=True)
model = model.to(device)

# ---------------------------
# 5. Train the Model on Training Scalar Fields
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
# criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))#
model.train()
num_epochs = 200

        # Measure training time
start_time = time.time()


for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # batch = batch.to(device)
        for key, item in batch:
            if torch.is_tensor(item):
                batch[key] = item.to(device)
        optimizer.zero_grad()
        logits = model(batch)  # Logits for all nodes in batch.
        loss = criterion(logits, batch.y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()  # Update the learning rate.
    if epoch % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


end_time = time.time()
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

# Save the trained model to a file.
model_save_path = "graph_nn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ---------------------------
# 6. Test Export (for Reference)
# ---------------------------
model.eval()
os.makedirs("test_fields_csv", exist_ok=True)    # CSV files for scalar fields (test data).
os.makedirs("extracted_points", exist_ok=True)     # CSV files for extracted critical points (test data).
os.makedirs("test_fields_obj", exist_ok=True)      # OBJ mesh files (test data).

correct = 0
total = 0

# Initialize counters for each class
num_classes = 4
correct_per_class = [0] * num_classes
total_per_class = [0] * num_classes

for idx, data_item in enumerate(test_data_list):
    with torch.no_grad():
        for key, item in data_item:
            if torch.is_tensor(item):
                data_item[key] = item.to(device)
        logits = model(data_item)
        predicted = logits.argmax(dim=1)  # Predicted class per node.
        for cls in range(num_classes):
            correct_per_class[cls] += ((predicted == cls) & (data_item.y == cls)).sum().item()
            total_per_class[cls] += (data_item.y == cls).sum().item()
        correct += (predicted == data_item.y.long()).sum().item()
        total += data_item.y.size(0)
        # save_data(data_item, idx,test_data_size,predicted)


#Accuracy = Number of correctly predicted nodes / Total number of nodes

accuracy = correct / total
print("Test Accuracy: {:.4f}".format(accuracy))

# Calculate and print accuracy for each class
for cls in range(num_classes):
    accuracy = correct_per_class[cls] / total_per_class[cls] if total_per_class[cls] > 0 else 0
    print(f"Class {cls} Accuracy: {accuracy:.4f}")




    # # (A) Export the test scalar field as a CSV file.
    # field = data_item.x.cpu().numpy()  
    # field_csv_filename = f"test_fields_csv/test_field_{idx}.csv"
    # np.savetxt(field_csv_filename, field, delimiter=",")
    # print(f"Test scalar field saved to {field_csv_filename}")
    
    # # (B) Extract predicted critical points.
    # critical_points = []
    # W = test_data_size[idx][0]
    # H = test_data_size[idx][1]
    # for i in range(predicted.shape[0]):
    #     val = predicted[i].item() if predicted[i].ndim == 0 else predicted[i].squeeze().item()
    #     if val < 3:
    #         row = i // W
    #         col = i % W
    #         critical_points.append((row, col, field[i], val))
    # points_csv_filename = f"extracted_points/extracted_points_{idx}.csv"
    # with open(points_csv_filename, mode="w", newline="") as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["x", "y", "scalar", "type"])
    #     csvwriter.writerows(critical_points)
    # print(f"Extracted critical points saved to {points_csv_filename}")
    
    # # (C) Export the test scalar field as an OBJ mesh using Paraview triangulation.
    # # Triangulation: For each cell (with vertices: top-left (v1), top-right (v2), bottom-right (v3), bottom-left (v4)):
    # #   - Triangle 1: (v1, v2, v4)
    # #   - Triangle 2: (v2, v3, v4)
    # obj_filename = f"test_fields_obj/test_field_{idx}.obj"
    # with open(obj_filename, mode="w") as f_obj:
    #     # Write vertices.
    #     for r in range(H):
    #         for c in range(W):
    #             x_coord = r      # row index
    #             y_coord = c      # column index
    #             z_coord = field[r*W+c]  # scalar value as z
    #             f_obj.write(f"v {x_coord} {y_coord} {z_coord}\n")
    #     # Write faces.
    #     for r in range(H - 1):
    #         for c in range(W - 1):
    #             v1 = r * W + c + 1        
    #             v2 = r * W + (c + 1) + 1     
    #             v3 = (r + 1) * W + (c + 1) + 1 
    #             v4 = (r + 1) * W + c + 1     
    #             # Triangle 2: top-right, bottom-right, bottom-left.
              
    #             f_obj.write(f"f {v1} {v2} {v3}\n")
    #             # Triangle 1: top-left, top-right, bottom-left.
    #             f_obj.write(f"f {v1} {v3} {v4}\n")
    # print(f"Test mesh OBJ saved to {obj_filename}")






# ---------------------------
# 7. Export the First 10 Training Fields as OBJ Meshes and Critical Points CSV
# ---------------------------
# os.makedirs("train_fields_obj", exist_ok=True)      # OBJ files for training fields.
# os.makedirs("extracted_points_train", exist_ok=True)   # CSV files for extracted points (training fields).

# for idx, data_item in enumerate(train_data_list[:10]):
#     field = data_item.x  # Numpy array (H, W)
    
#     # (A) Export the training scalar field as an OBJ mesh directly.
#     obj_filename = f"train_fields_obj/train_field_{idx}.obj"
#     with open(obj_filename, mode="w") as f_obj:
#         # Write vertices.
#         for r in range(H):
#             for c in range(W):
#                 x_coord = c
#                 y_coord = r
#                 z_coord = field[c*W+r]  # fixed: use row then col
#                 f_obj.write(f"v {x_coord} {y_coord} {z_coord}\n")
#         # Write faces.
#         for r in range(H - 1):
#             for c in range(W - 1):
#                 v1 = r * W + c + 1         
#                 v2 = r * W + (c + 1) + 1     
#                 v3 = (r + 1) * W + (c + 1) + 1 
#                 v4 = (r + 1) * W + c + 1    
#                 f_obj.write(f"f {v1} {v2} {v3}\n")
#                 f_obj.write(f"f {v1} {v3} {v4}\n")
#     print(f"Train mesh OBJ saved to {obj_filename}")
    
    # # (B) Directly extract the critical points from the ground truth labels in data_item.y.
    # #    (No model inference is performed here.)
    # pred = data_item.y
    # critical_points = []
    # for i in range(pred.shape[0]):
    #     if pred[i].item() == 1:
    #         row = i // W
    #         col = i % W
    #         critical_points.append((row, col, field[row*W+col]))
    # points_csv_filename = f"extracted_points_train/extracted_points_{idx}.csv"
    # with open(points_csv_filename, mode="w", newline="") as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["x", "y", "scalar"])
    #     csvwriter.writerows(critical_points)
    # print(f"Train critical points saved to {points_csv_filename}")