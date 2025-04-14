import torch as th
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv

# GraphConv-based Node Classifier
class GraphNodeClassifier(th.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        channels1=16
        channels2=32
        channels3=64

        self.conv1 = GraphConv(in_channels, channels1)
        self.Linear1 = Linear(channels1,channels2)
        self.conv2 = GraphConv(channels2,channels3)
        self.Linear2 = Linear(channels3,out_channels)

    def forward(self, x, edge_index):
        x = x.reshape((-1,1))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.Linear2(x)
        return x