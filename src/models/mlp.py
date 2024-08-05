import torch.nn as nn
import torch
import math

from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int = 3, dropout: [int]  = [], prob = 0.5, dynamic_creation = True) -> None:
        super(MLP, self).__init__()
        layers = OrderedDict()

        if dynamic_creation:
            # Linearly spaced linear layers
            step = math.ceil((in_size-out_size)/num_layers)
            in_features = in_size
            for i in range(num_layers - 1):

                layers['lin_{}'.format(i)] = nn.Linear(in_features, in_features - step)
                if i != num_layers-1:
                    layers['relu_{}'.format(i)] = nn.ReLU()
                if i in dropout:
                    layers['drop_{}'.format(i)] = nn.Dropout(p=prob)
                in_features -= step
            layers['lin_{}'.format(num_layers-1)] = nn.Linear(in_features, out_size)
        else:
            layers['lin_{}'.format(0)] = nn.Linear(in_size, 256)
            layers['relu_{}'.format(0)] = nn.ReLU()
            layers['drop_{}'.format(0)] = nn.Dropout(p=prob)
            layers['lin_{}'.format(1)] = nn.Linear(256, 256)
            layers['relu_{}'.format(1)] = nn.ReLU()
            layers['drop_{}'.format(1)] = nn.Dropout(p=prob)
            layers['lin_{}'.format(2)] = nn.Linear(256, out_size)

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = torch.squeeze(x)
        return self.net(x)