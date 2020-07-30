from __future__ import print_function

import torch
import torch.nn as nn
from models.resnet import InsResNet50
import torch.nn.functional as F
import pdb

# Encoder
# Resnet with MLP projection head


class simCLR_encoder(nn.Module):
    def __init__(self, feature_dim=128, in_channel=1):
        super(simCLR_encoder, self).__init__()

        # localInfoMax
        self.f = nn.Sequential(*list(InsResNet50().encoder.module.children())[:-3])
        self.f[0] = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # h from simCLR
        self.pool = None
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        M = self.f(x)
        self.pool = nn.AvgPool2d(kernel_size=M.size(2), stride=1, padding=0)
        x = self.pool(M)
        feature = torch.flatten(x, start_dim=1)
        # return F.normalize(feature, dim=-1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), M


class simCLR(nn.Module):
    """Encoder for instance discrimination and MoCo"""

    def __init__(self, feature_dim=128, in_channel=1):
        super(simCLR, self).__init__()
        self.encoder = simCLR_encoder(feature_dim, in_channel)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x):
        return self.encoder(x)
