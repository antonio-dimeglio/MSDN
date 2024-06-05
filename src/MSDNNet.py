import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np

class MSDNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_features, dilations):
        super(MSDNet, self).__init__()
        self.num_layers = num_layers
        self.dilations = dilations

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=dilations[0], dilation=dilations[0]))

        for i in range(1, num_layers):
            self.layers.append(nn.Conv2d(num_features * (i+1), num_features, kernel_size=3, padding=dilations[i], dilation=dilations[i]))

        self.output_layer = nn.Conv2d(num_features * num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        outputs = [x]
        for i in range(self.num_layers):
            out = torch.cat(outputs, dim=0)
            out = self.layers[i](out)
            out = F.relu(out)
            outputs.append(out)
        
        out = torch.cat(outputs[1:], dim=0)
        out = self.output_layer(out)
        return out
