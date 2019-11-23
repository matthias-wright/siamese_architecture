
"""
This file contains the implementations for all of the networks used in the siamese architecture.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Implements a residual block as proposed by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, groups=1):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True,
                                           track_running_stats=True)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=groups, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True,
                                           track_running_stats=True)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                          dilation=dilation, groups=groups, bias=False),
                nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.9, affine=True, track_running_stats=True)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class Map(nn.Module):
    """
    Implements the network architecture that maps between the low-dimensional representations.
    Referred to as 'transformation networks' or 'generators' in the thesis. We used the same architecture
    for both domains.
    """

    def __init__(self):
        super(Map, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class Discriminator(nn.Module):
    """
    Implements a discriminator network. We used the same architecture for both domains.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 128)
        out = self.linear(out)
        return out


class Interpolate(nn.Module):
    """
    Downsamples or upsamples the input according to the given scale factor.
    """
    def __init__(self, scale_factor, mode):
        """
        :param scale_factor: (float) the factor by which the input is downsampled / upsampled.
        :param mode: the method that is used for sampling. Possible ptions: 'nearest', 'linear',
                     'bilinear', 'bicubic', 'trilinear', 'area'.
        """
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class AutoEncoder(nn.Module):
    """
    Implements an Autoencoder network. We used the same architecture for both domains.
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=4, stride=2),
            nn.LeakyReLU(),
        )

        self.decode = nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            Interpolate(scale_factor=2, mode='nearest'),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def encoder(self, x):
        out = self.encode(x)
        return out

    def decoder(self, z):
        out = self.decode(z)
        return out

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return r