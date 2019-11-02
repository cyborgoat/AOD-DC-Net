"""Created by Shuhao Ren at 11/1/19"""

import torch
from torch.nn import *


class AODNet(Module):

    def __init__(self):
        super(AODNet, self).__init__()
        self.conv_1 = Conv2d(3, 3, 1, 1, 0, bias=True)
        self.conv_2 = Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv_3 = Conv2d(6, 3, 5, 1, 2, bias=True)
        self.conv_4 = Conv2d(6, 3, 7, 1, 3, bias=True)
        self.conv_5 = Conv2d(12, 3, 3, 1, 1, bias=True)
        self.relu = ReLU(inplace=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_1 = self.relu(self.conv_1(x))
        x_2 = self.relu(self.conv_2(x_1))
        concat_1 = torch.cat((x_1, x_2), 1)
        x_3 = self.relu(self.conv_3(concat_1))
        concat_2 = torch.cat((x_2, x_3), 1)
        x_4 = self.relu(self.conv_4(concat_2))
        concat_3 = torch.cat((x_1, x_2, x_3, x_4), 1)
        x_5 = self.relu(self.conv_5(concat_3))
        return self.relu((x_5 * x) - x_5 + 1)
