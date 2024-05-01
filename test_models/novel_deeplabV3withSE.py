import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any
from typing import List

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UNetDeeplab_SE(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNetDeeplab_SE, self).__init__()

        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        if input_channel != 3:
            self.model.backbone.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.classifier[4] = nn.Conv2d(256, output_channel, kernel_size=1, stride=1)

        for m in self.model.backbone.modules():
            if isinstance(m, models.resnet.Bottleneck):
                m.conv3 = nn.Sequential(m.conv3, SELayer(m.conv3.out_channels))

    def forward(self, x):
        features = self.model(x)['out']
        return features