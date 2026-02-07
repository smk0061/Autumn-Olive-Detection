# unet_model.py
# u-net architecture for semantic segmentation of uas imagery.
# shared module imported by unet_training.py and unet_prediction.py.

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = activation()
        self.dropout1 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = activation()
        self.dropout2 = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size, activation, dropout_rate=0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv_block = DoubleConv(skip_channels + out_channels, out_channels, kernel_size, activation, dropout_rate=dropout_rate)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, activation, num_classes, dropout_rate=0.3):
        super().__init__()
        self.down1 = DoubleConv(in_channels, num_filters, kernel_size, activation, dropout_rate=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(num_filters, num_filters * 2, kernel_size, activation, dropout_rate=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = DoubleConv(num_filters * 2, num_filters * 4, kernel_size, activation, dropout_rate=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = DoubleConv(num_filters * 4, num_filters * 8, kernel_size, activation, dropout_rate=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(num_filters * 8, num_filters * 16, kernel_size, activation, dropout_rate=dropout_rate)

        self.up1 = DecoderBlock(num_filters * 16, num_filters * 8, num_filters * 8, kernel_size, activation, dropout_rate=0.2)
        self.up2 = DecoderBlock(num_filters * 8, num_filters * 4, num_filters * 4, kernel_size, activation, dropout_rate=0.2)
        self.up3 = DecoderBlock(num_filters * 4, num_filters * 2, num_filters * 2, kernel_size, activation, dropout_rate=0.1)
        self.up4 = DecoderBlock(num_filters * 2, num_filters, num_filters, kernel_size, activation, dropout_rate=0.1)

        self.final_conv = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        u1 = self.up1(bn, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final_conv(u4)


def get_input_channels(model_type):
    if model_type == 'rgb':
        return 3
    elif model_type == 'multispectral':
        return 5
    elif model_type == 'vi_composite':
        return 8
    else:
        raise ValueError(f"unknown model_type: {model_type}")


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
