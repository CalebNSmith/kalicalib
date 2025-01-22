# File: src/models/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .layers import NLBlockND

class DilatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(DilatedBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class KaliCalibNet(nn.Module):
    def __init__(self, n_keypoints):
        super(KaliCalibNet, self).__init__()

        # Load pretrained ResNet-18
        print(f"Initializing KaliCalibNet with {n_keypoints} channels ({n_keypoints-2} grid points + ub + lb)")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("Loaded pretrained ResNet-18")

        # Initial layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = self._make_dilated_layer(resnet.layer3, 2, 128, 256, stride=2)
        self.layer4 = self._make_dilated_layer(resnet.layer4, 4, 256, 512, stride=2)

        # Non-local blocks
        self.non_local3 = NLBlockND(256, dimension=2)  # Changed to NLBlockND and added dimension
        self.non_local4 = NLBlockND(512, dimension=2)  # Changed to NLBlockND and added dimension

        # Decoder layers - Using output_padding to ensure exact size matching
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, n_keypoints + 1, kernel_size=1, stride=1, padding=0)
        )

    def _make_dilated_layer(self, layer, dilation, in_channels, out_channels, stride=1):
        """
        Convert a regular ResNet layer to use dilated convolutions
        using the DilatedBasicBlock.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels * DilatedBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * DilatedBasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * DilatedBasicBlock.expansion),
            )

        dilated_layer = nn.Sequential(
            DilatedBasicBlock(in_channels, out_channels, stride=stride, dilation=dilation, downsample=downsample),
            DilatedBasicBlock(out_channels, out_channels, dilation=dilation)
        )
        return dilated_layer

    def forward(self, x):
        # Initial convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder path with skip connections
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e3 = self.non_local3(e3)
        e4 = self.layer4(e3)
        e4 = self.non_local4(e4)

        # Decoder path with skip connections
        d1 = self.decoder1(e4)
        if d1.shape[2:] != e3.shape[2:]:
            d1 = F.interpolate(d1, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d1_cat = torch.cat([d1, e3], dim=1)
        d2 = self.decoder2(d1_cat)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2_cat = torch.cat([d2, e2], dim=1)
        d3 = self.decoder3(d2_cat)
        if d3.shape[2:] != e1.shape[2:]:
            d3 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d3_cat = torch.cat([d3, e1], dim=1)
        d4 = self.decoder4(d3_cat)

        # Final softmax
        out = F.softmax(d4, dim=1)

        return out