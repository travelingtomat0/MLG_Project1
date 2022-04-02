import numpy as np
import torch
from torch import nn

from collections import OrderedDict


class ConvolutionalModel(nn.Module):

    def __init__(self, c=10, d=None, k=None, window_size=50000, batch_size=1):
        super(ConvolutionalModel, self).__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ('Conv1', nn.Conv1d(in_channels=c, out_channels=5, kernel_size=10, stride=10)),
                ('ReLU1', nn.ReLU()),
                ('MaxPool1', nn.MaxPool1d(kernel_size=10)),
                ('Conv2', nn.Conv1d(in_channels=5, out_channels=1, kernel_size=10, stride=10)),
                ('ReLU2', nn.ReLU()),
                ('MaxPool2', nn.MaxPool1d(kernel_size=10)),
                ('Linear', nn.Linear(in_features=10, out_features=1)),
                ('Final Activation', nn.ReLU())
            ])
        )

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        return self.layers(x)[0]
