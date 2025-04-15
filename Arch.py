import torch as th
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv

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

class HybridClassifier(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1=16
        channels2=32
        channels3=64

        self.CPConv1 = GraphConv(1, channels1)
        self.CPLinear1 = Linear(channels1,channels2)
        self.CPConv2 = GraphConv(channels2,channels3)
        self.CPLinear2 = Linear(channels3,2)

        self.CTConv1 = GraphConv(1, channels1)
        self.CTLinear1 = Linear(channels1,channels2)
        self.CTConv2 = GraphConv(channels2,channels3)
        self.CTLinear2 = Linear(channels3,3)

    def forward(self, x, edge_index):
        x = x.reshape((-1,1))

        xcp = self.CPConv1(x, edge_index)
        xcp = F.relu(xcp)
        xcp = self.CPLinear1(xcp)
        xcp = F.relu(xcp)
        xcp = self.CPConv2(xcp, edge_index)
        xcp = F.relu(xcp)
        xcp = self.CPLinear2(xcp)

        xct = self.CTConv1(x, edge_index)
        xct = F.relu(xct)
        xct = self.CTLinear1(xct)
        xct = F.relu(xct)
        xct = self.CTConv2(xct, edge_index)
        xct = F.relu(xct)
        xct = self.CTLinear2(xct)
        zero_column = th.zeros(xct.shape[0], 1)
        xct = th.cat((zero_column, xct),dim=1)

        return xcp, xct
    
class HybridClassifierLarge(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1=16
        channels2=32
        channels3=64

        self.NormalConv1 = GraphConv(1, channels1)
        self.NormalLinear1 = Linear(channels1,channels2)
        self.NormalConv2 = GraphConv(channels2,channels3)
        self.NormalLinear2 = Linear(channels3,2)

        self.MaxConv1 = GraphConv(1, channels1)
        self.MaxLinear1 = Linear(channels1,channels2)
        self.MaxConv2 = GraphConv(channels2,channels3)
        self.MaxLinear2 = Linear(channels3,2)

        self.MinConv1 = GraphConv(1, channels1)
        self.MinLinear1 = Linear(channels1,channels2)
        self.MinConv2 = GraphConv(channels2,channels3)
        self.MinLinear2 = Linear(channels3,2)

    def forward(self, x, edge_index):
        x = x.reshape((-1,1))

        xn = self.NormalConv1(x, edge_index)
        xn = F.relu(xn)
        xn = self.NormalLinear1(xn)
        xn = F.relu(xn)
        xn = self.NormalConv2(xn, edge_index)
        xn = F.relu(xn)
        xn = self.NormalLinear2(xn)

        xma = self.MaxConv1(x, edge_index)
        xma = F.relu(xma)
        xma = self.MaxLinear1(xma)
        xma = F.relu(xma)
        xma = self.MaxConv2(xma, edge_index)
        xma = F.relu(xma)
        xma = self.MaxLinear2(xma)

        xmi = self.MinConv1(x, edge_index)
        xmi = F.relu(xmi)
        xmi = self.MinLinear1(xmi)
        xmi = F.relu(xmi)
        xmi = self.MinConv2(xmi, edge_index)
        xmi = F.relu(xmi)
        xmi = self.MinLinear2(xmi)

        return xn, xma, xmi
    
class inplaceCNN(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1 = 16
        channels2 = 32
        channels3 = 64

        self.conv1 = Conv2d(1, channels1, kernel_size=3, padding=1)
        self.conv2 = Conv2d(channels1, channels2, kernel_size=1)
        self.conv3 = Conv2d(channels2, channels3, kernel_size=1)
        self.conv4 = Conv2d(channels3, 4, kernel_size=1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)

        C,W,H = x.shape
        x = x.reshape((C,W*H))
        x = th.permute(x, (1,0))

        return x
    
class inplaceCNN2(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1 = 32
        channels2 = 64
        channels3 = 128

        self.conv1 = Conv2d(1, channels1, kernel_size=3, padding=1)
        self.conv2 = Conv2d(channels1, channels2, kernel_size=1)
        self.conv3 = Conv2d(channels2, channels3, kernel_size=1)
        self.conv4 = Conv2d(channels3, 4, kernel_size=1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)

        C,W,H = x.shape
        x = x.reshape((C,W*H))
        x = th.permute(x, (1,0))

        return x
    
class inplaceCNN2Binary(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1 = 32
        channels2 = 64
        channels3 = 128

        self.conv1 = Conv2d(1, channels1, kernel_size=3, padding=1)
        self.conv2 = Conv2d(channels1, channels2, kernel_size=1)
        self.conv3 = Conv2d(channels2, channels3, kernel_size=1)
        self.conv4 = Conv2d(channels3, 2, kernel_size=1)

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)

        C,W,H = x.shape
        x = x.reshape((C,W*H))
        x = th.permute(x, (1,0))

        return x

class inplaceCNNTwoLevel(th.nn.Module):
    def __init__(self):
        super().__init__()

        channels1 = 32
        channels2 = 64
        channels3 = 128

        self.CPconv1 = Conv2d(1, channels1, kernel_size=3, padding=1)
        self.CPconv2 = Conv2d(channels1, channels2, kernel_size=1)
        self.CPconv3 = Conv2d(channels2, channels3, kernel_size=1)
        self.CPconv4 = Conv2d(channels3, 2, kernel_size=1)

        self.CTconv1 = Conv2d(1, channels1, kernel_size=3, padding=1)
        self.CTconv2 = Conv2d(channels1, channels2, kernel_size=1)
        self.CTconv3 = Conv2d(channels2, channels3, kernel_size=1)
        self.CTconv4 = Conv2d(channels3, 4, kernel_size=1)

    def forward(self,x):
        xcp = self.CPconv1(x)
        xcp = F.leaky_relu(xcp)
        xcp = self.CPconv2(xcp)
        xcp = F.leaky_relu(xcp)
        xcp = self.CPconv3(xcp)
        xcp = F.leaky_relu(xcp)
        xcp = self.CPconv4(xcp)

        C,W,H = xcp.shape
        xcp = xcp.reshape((C,W*H))
        xcp = th.permute(xcp, (1,0))

        xct = self.CTconv1(x)
        xct = F.leaky_relu(xct)
        xct = self.CTconv2(xct)
        xct = F.leaky_relu(xct)
        xct = self.CTconv3(xct)
        xct = F.leaky_relu(xct)
        xct = self.CTconv4(xct)

        C,W,H = xct.shape
        xct = xct.reshape((C,W*H))
        xct = th.permute(xct, (1,0))

        return xcp, xct