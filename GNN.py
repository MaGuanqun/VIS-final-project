import torch
import torch.nn.functional as F
import numpy as np
import csv
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ---------------------------
# 1. Prepare the grid and graph data
# ---------------------------
H, W = 10, 10  # grid dimensions
# Create a synthetic scalar field (replace with your actual data if needed)
field = np.random.rand(H, W)

# Prepare node features: each node gets the scalar field value as a feature.
x = torch.tensor(field.reshape(-1, 1), dtype=torch.float)

# Create edge connectivity (4-neighbor connectivity)
edges = []
for i in range(H):
    for j in range(W):
        current_node = i * W + j
        # Up
        if i > 0:
            edges.append((current_node, (i - 1) * W + j))
        # Down
        if i < H - 1:
            edges.append((current_node, (i + 1) * W + j))
        # Left
        if j > 0:
            edges.append((current_node, i * W + (j - 1)))
        # Right
        if j < W - 1:
            edges.append((current_node, i * W + (j + 1)))
# Convert edge list to tensor with shape [2, num_edges]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Generate synthetic labels: a node is labeled as critical (1) if its scalar value is larger than its available neighbors.
y = []
for i in range(H):
    for j in range(W):
        current_value = field[i, j]
        neighbor_values = []
        if i > 0: neighbor_values.append(field[i - 1, j])
        if i < H - 1: neighbor_values.append(field[i + 1, j])
        if j > 0: neighbor_values.append(field[i, j - 1])
        if j < W - 1: neighbor_values.append(field[i, j + 1])
        is_critical = 1 if current_value > max(neighbor_values) else 0
        y.append(is_critical)
y = torch.tensor(y, dtype=torch.long)

# Create the PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y)

# ---------------------------
# 2. Define the GNN Model (using GCNConv)
# ---------------------------
class GraphNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Log probabilities for 2 classes (critical/not-critical)
        return F.log_softmax(x, dim=1)

# Instantiate the model
model = GraphNN(in_channels=1, hidden_channels=16, out_channels=2)

# ---------------------------
# 3. Train the Model (Dummy Training Loop)
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ---------------------------
# 4. Evaluation: Extract Critical Points
# ---------------------------
model.eval()
with torch.no_grad():
    out = model(data)
    # Get predicted class for each node (0: non-critical, 1: critical)
    pred = out.argmax(dim=1)

# Collect the extracted points (only nodes predicted as critical)
extracted_points = []  # list for csv export; each entry will be (x, y, scalar_value)
# Here we compute the grid coordinates from the 1D node index.
for idx in range(pred.shape[0]):
    if pred[idx] == 1:
        i = idx // W  # row index
        j = idx % W   # column index
        extracted_points.append((j, i, field[i, j]))  # (x, y, scalar)

# Write the extracted points to CSV.
csv_filename = 'extracted_points.csv'
with open(csv_filename, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow(['x', 'y', 'scalar'])
    # Write data rows
    csvwriter.writerows(extracted_points)
print(f"Extracted points saved to {csv_filename}")

# ---------------------------
# 5. Save the Mesh as an OBJ File
# ---------------------------
# We construct a mesh over the grid. Each grid point becomes a vertex.
# Here, we set the vertex coordinates as (j, field[i,j], i) so that the scalar is mapped to the y-axis.
obj_filename = 'mesh.obj'
with open(obj_filename, mode='w') as f:
    # Write vertices
    for i in range(H):
        for j in range(W):
            # Note: Adjust coordinate mapping as needed.
            x_coord = j
            y_coord = field[i, j]
            z_coord = i
            f.write(f"v {x_coord} {y_coord} {z_coord}\n")
    # Write faces: each cell in the grid is split into two triangles.
    # Be careful: OBJ vertex indices are 1-indexed.
    for i in range(H - 1):
        for j in range(W - 1):
            v1 = i * W + j + 1
            v2 = i * W + (j + 1) + 1
            v3 = (i + 1) * W + j + 1
            v4 = (i + 1) * W + (j + 1) + 1
            # First triangle (v1, v2, v3)
            f.write(f"f {v1} {v2} {v3}\n")
            # Second triangle (v2, v4, v3)
            f.write(f"f {v2} {v4} {v3}\n")
print(f"Mesh saved to {obj_filename}")