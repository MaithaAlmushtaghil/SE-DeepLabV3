import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt

class unetplusplus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetplusplus, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class NestedUNet(nn.Module):
    def __init__(self, input_channel, output_channel, features=[64, 128, 256, 512]):
        super(NestedUNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for feat in features: #Initialize the encoder
            self.encoder.append(
                unetplusplus(input_channel, feat)
            )
            input_channel = feat

        for feat in reversed(features): 
            self.decoder.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.decoder.append(unetplusplus(feat * 2, feat))
            self.attention_blocks.append(AttentionGate(F_g=feat, F_l=feat, F_int=feat // 2))

        self.bottleneck = unetplusplus(features[-1], features[-1] * 2)
        self.final_layer = nn.Conv2d(features[0], output_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layers in self.encoder:
            x = layers(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        att_idx = 0

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            att = self.attention_blocks[att_idx] 
            att_idx += 1
            x = att(skip_connection, x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_layer(x)