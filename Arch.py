import torch as th
import torch.nn.functional as F
from torch.nn import Conv2d

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

        xct = self.CTconv1(x)
        xct = F.leaky_relu(xct)
        xct = self.CTconv2(xct)
        xct = F.leaky_relu(xct)
        xct = self.CTconv3(xct)
        xct = F.leaky_relu(xct)
        xct = self.CTconv4(xct)

        return xcp, xct