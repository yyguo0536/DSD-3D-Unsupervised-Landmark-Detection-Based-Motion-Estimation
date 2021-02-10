from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from modules.vnet_3d import Temporal_Encoder, Temporal_Decoder, Temporal_Encoder_D, Temporal_Decoder_D
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']#.to(device0)

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()).to(device0)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    zz = z.view(-1, 1, 1).repeat(1, h, w)
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out



class SameBlock3d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        #out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, in_chal=1, out_chal=32):
        super(Hourglass, self).__init__()
        self.encoder = Temporal_Encoder(in_chal)
        self.decoder = Temporal_Decoder(out_chal)
        self.out_filters = out_chal

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Hourglass_D(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, in_chal=1, out_chal=32):
        super(Hourglass_D, self).__init__()
        self.encoder = Temporal_Encoder_D(in_chal)
        self.decoder = Temporal_Decoder_D(out_chal)
        self.out_filters = out_chal

    def forward(self, x):
        return self.decoder(self.encoder(x))



class Hourglass_ND(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, in_chal=1, out_chal=32):
        super(Hourglass_ND, self).__init__()
        self.encoder = Temporal_Encoder_ND(in_chal)
        self.decoder = Temporal_Decoder_ND(out_chal)
        self.out_filters = out_chal

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AntiAliasInterpolation3d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation3d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        self.kc = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size, kernel_size]
        sigma = [sigma, sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.kc, self.ka, self.kb, self.kc))
        out = F.conv3d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale, self.scale))

        return out
