import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import os
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# Set random seeds for reproducibility.
# np.random.seed(42)
torch.manual_seed(42)

# ---------------------------
# Parameters
# ---------------------------
H, W = 10, 10  # Grid dimensions

# ---------------------------
# 1. Build Triangulation Edge Index
# ---------------------------
def generate_triangulation_edge_index(H, W):
    """
    For a grid of size H x W, generate edges based on triangulation.
    For each cell (with vertices: top-left, top-right, bottom-right, bottom-left),
    we use the following triangulation:
      - Triangle 1: top-left, top-right, bottom-left
      - Triangle 2: top-right, bottom-right, bottom-left
    We then also add direct horizontal and vertical edges so that boundary
    connectivity is complete.
    Returns a tensor of shape [2, E].
    """
    edges = set()
    # Process each cell (for interior cells).
    for i in range(H - 1):
        for j in range(W- 1):
            v1 = i * W + j        
            v2 = i * W + (j + 1)    
            v3 = (i + 1) * W + (j + 1)  
            v4 = (i + 1) * W + j    
            # Triangle 1: v1, v2, v4
            edges.add((v1, v2)); edges.add((v2, v1))
            edges.add((v1, v3)); edges.add((v3, v1))
            edges.add((v1, v4)); edges.add((v4, v1))
            # Triangle 2: v2, v3, v4
            edges.add((v2, v3)); edges.add((v3, v2))
            edges.add((v3, v4)); edges.add((v4, v3))
            # (Edge (v2,v4) already added above.)
    # # Also add horizontal edges along each row.
    # for i in range(H):
    #     for j in range(W - 1):
    #         v1 = i * W + j
    #         v2 = i * W + (j + 1)
    #         edges.add((v1, v2)); edges.add((v2, v1))
    # # Also add vertical edges along each column.
    # for i in range(H - 1):
    #     for j in range(W):
    #         v1 = i * W + j
    #         v2 = (i + 1) * W + j
    #         edges.add((v1, v2)); edges.add((v2, v1))
    # Deduplicate edges by treating (u, v) and (v, u) as the same.
    edge_list = list(edges)
    edge_list = [tuple(sorted(edge)) for edge in edge_list]  # Ensure consistent ordering of edges
    edge_list = list(set(edge_list))  # Deduplicate edges
    edge_list.sort()  # Sort edges for consistent indexing
    # print(edge_list)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def generate_neighbor_dict(edge_index):
    """
    Given an edge_index tensor of shape [2, E] (with undirected edges),
    return a dictionary mapping each vertex index to a set of neighbor vertex indices.
    """
    neighbor_dict = {}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        if u not in neighbor_dict:
            neighbor_dict[u] = set()
        if v not in neighbor_dict:
            neighbor_dict[v] = set()
        neighbor_dict[u].add(v)
        neighbor_dict[v].add(u)
    return neighbor_dict

# Generate triangulation connectivity and neighbor dictionary.
tri_edge_index = generate_triangulation_edge_index(H, W)
neighbor_dict = generate_neighbor_dict(tri_edge_index)

# ---------------------------
# 2. Utility: Create a PyG Data Object from a Scalar Field Using Triangulation Neighbors
# ---------------------------
def create_field_data(field):
    """
    Given a numpy scalar field of shape (H, W), this function returns a PyTorch Geometric
    Data object that uses the triangulation connectivity.
      - x: node features (the scalar value at each grid point, as a column vector)
      - y: binary labels for each node. A node is labeled as critical (1) if its scalar value is strictly 
           greater than the scalar values at all neighbors determined by the triangulation connectivity.
      - edge_index: the triangulation edge index computed above.
    """
    # Node features.
    x = torch.tensor(field.reshape(-1, 1), dtype=torch.float)
    y_list = []
    num_vertices = H * W
    for i in range(num_vertices):
        current_val = field[i // H, i % H]
        # Get neighbors from the neighbor dictionary; if no neighbor found, use empty list.
        neighbors = neighbor_dict.get(i, [])

        # print(f"Node {i}: current_val={current_val}, neighbors={neighbors}")
        if neighbors:
            neighbor_vals = [field[n // H, n % H] for n in neighbors]
            # print(current_val, neighbor_vals)
            
            is_critical = 1 if current_val > max(neighbor_vals) else 0
        else:
            is_critical = 0
        y_list.append(is_critical)
    y = torch.tensor(y_list, dtype=torch.float)
    return Data(x=x, edge_index=tri_edge_index, y=y)

# ---------------------------
# 3. Generate Training and Test Data
# ---------------------------
num_train = 50  # Number of training scalar fields.
num_test = 1   # Number of test scalar fields.

train_data_list = []
test_data_list = []

# Generate training fields.
for idx in range(num_train):
    field = np.random.rand(H, W)
    data_obj = create_field_data(field)
    data_obj.field = field  # Save the original field for exporting.
    train_data_list.append(data_obj)

# Generate test fields.
for idx in range(num_test):
    field = np.random.rand(H, W)
    data_obj = create_field_data(field)
    data_obj.field = field
    test_data_list.append(data_obj)

# Compute the positive weight from the training data to mitigate class imbalance.
total_pos, total_count = 0, 0
for data in train_data_list:
    total_pos += data.y.sum().item()
    total_count += data.y.shape[0]
total_neg = total_count - total_pos
pos_weight_value = total_neg / (total_pos + 1e-8)  # Avoid division by zero.
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float)
print(f"Training samples: {total_count}, Positive: {total_pos}, Negative: {total_neg}, pos_weight: {pos_weight.item():.2f}")

# Create DataLoaders.
train_loader = DataLoader(train_data_list, batch_size=10, shuffle=True)
# We process test (and later training export) one graph at a time.
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)

# ---------------------------
# 4. Define the Binary Classification GNN Model
# ---------------------------
class GraphNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_p=0.3, use_residual=False):
        """
        Modified network structure with:
          - Three GCNConv layers to increase depth.
          - Batch Normalization after the first two layers.
          - Dropout for regularization.
          - Optionally adding a residual connection from the input.
          
        Parameters:
          - in_channels: Number of input features (1 in this case).
          - hidden_channels: Number of hidden units (e.g., 16, 32, 64, etc.).
          - dropout_p: Dropout probability.
          - use_residual: If True, adds a skip connection from the input features to the output.
        """
        super(GraphNN, self).__init__()
        self.use_residual = use_residual

        # First Graph Convolution layer.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Second layer increases the model depth.
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Third layer outputs a single logit per node.
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        # # Optional residual connection: project input features to match output size.
        if use_residual:
            self.res_fc = nn.Linear(in_channels, 1)
        
        # Dropout parameter to help regularize the network.
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        h = self.dropout(h)
        h = self.conv4(h, edge_index)
        if self.use_residual:
            # Directly projects x to one logit per node and adds it.
            h = h + self.res_fc(x)
        return h.view(-1)
        
        # # Optionally add a residual (skip) connection.

        
        # Flatten the output to 1D per node.

model = GraphNN(in_channels=1, hidden_channels=32, dropout_p=0.3, use_residual=True)

# ---------------------------
# 5. Train the Model on Training Scalar Fields
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model.train()
num_epochs = 600
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch)  # Logits for all nodes in batch.
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()  # Update the learning rate.
    if epoch % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# ---------------------------
# 6. Test Export (for Reference)
# ---------------------------
model.eval()
os.makedirs("test_fields_csv", exist_ok=True)    # CSV files for scalar fields (test data).
os.makedirs("extracted_points", exist_ok=True)     # CSV files for extracted critical points (test data).
os.makedirs("test_fields_obj", exist_ok=True)      # OBJ mesh files (test data).

for idx, data_item in enumerate(test_data_list):
    with torch.no_grad():
        logits = model(data_item)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
    
    # (A) Export the test scalar field as a CSV file.
    field = data_item.field  # Numpy array of shape (H, W)
    field_csv_filename = f"test_fields_csv/test_field_{idx}.csv"
    np.savetxt(field_csv_filename, field, delimiter=",")
    print(f"Test scalar field saved to {field_csv_filename}")
    
    # (B) Extract predicted critical points.
    critical_points = []
    for i in range(pred.shape[0]):
        val = pred[i].item() if pred[i].ndim == 0 else pred[i].squeeze().item()
        if val == 1:
            row = i // W
            col = i % W
            critical_points.append((row, col, field[row, col]))
    points_csv_filename = f"extracted_points/extracted_points_{idx}.csv"
    with open(points_csv_filename, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["x", "y", "scalar"])
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
                x_coord = c      # column index
                y_coord = r      # row index
                z_coord = field[c, r]  # scalar value as z
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

# ---------------------------
# 7. Export the First 10 Training Fields as OBJ Meshes and Critical Points CSV
# ---------------------------
os.makedirs("train_fields_obj", exist_ok=True)      # OBJ files for training fields.
os.makedirs("extracted_points_train", exist_ok=True)   # CSV files for extracted points (training fields).

for idx, data_item in enumerate(train_data_list[:10]):
    field = data_item.field  # Numpy array (H, W)
    
    # (A) Export the training scalar field as an OBJ mesh directly.
    obj_filename = f"train_fields_obj/train_field_{idx}.obj"
    with open(obj_filename, mode="w") as f_obj:
        # Write vertices.
        for r in range(H):
            for c in range(W):
                x_coord = c
                y_coord = r
                z_coord = field[c, r]  # fixed: use row then col
                f_obj.write(f"v {x_coord} {y_coord} {z_coord}\n")
        # Write faces.
        for r in range(H - 1):
            for c in range(W - 1):
                v1 = r * W + c + 1         
                v2 = r * W + (c + 1) + 1     
                v3 = (r + 1) * W + (c + 1) + 1 
                v4 = (r + 1) * W + c + 1    
                f_obj.write(f"f {v1} {v2} {v3}\n")
                f_obj.write(f"f {v1} {v3} {v4}\n")
    print(f"Train mesh OBJ saved to {obj_filename}")
    
    # (B) Directly extract the critical points from the ground truth labels in data_item.y.
    #    (No model inference is performed here.)
    pred = data_item.y
    critical_points = []
    for i in range(pred.shape[0]):
        if pred[i].item() == 1:
            row = i // W
            col = i % W
            critical_points.append((row, col, field[row, col]))
    points_csv_filename = f"extracted_points_train/extracted_points_{idx}.csv"
    with open(points_csv_filename, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["x", "y", "scalar"])
        csvwriter.writerows(critical_points)
    print(f"Train critical points saved to {points_csv_filename}")