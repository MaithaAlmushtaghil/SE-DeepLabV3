import torchvision.models as models
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

class UNetDeeplab101(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNetDeeplab101, self).__init__()

        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        if input_channel != 3: 
            self.model.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.classifier[4] = nn.Conv2d(256, output_channel, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']