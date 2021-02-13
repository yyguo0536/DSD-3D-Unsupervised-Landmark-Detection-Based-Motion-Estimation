from torch import nn
import torch
import torch.nn.functional as F
from modules.util_3d import Hourglass, make_coordinate_grid, AntiAliasInterpolation3d
import fractions
from gru_net import *

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, motion_num_kp, still_num_kp,  
                num_channels, max_features,
                 num_blocks, temperature, lvm_num, lv_num, rv_num, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=1):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(num_channels, motion_num_kp+still_num_kp)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=motion_num_kp+still_num_kp, kernel_size=(3, 3, 3),
                            padding=pad)


        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else motion_num_kp+still_num_kp
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters,
                                      out_channels=9 * self.num_jacobian_maps, kernel_size=(3, 3, 3), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = float(fractions.Fraction(scale_factor[0],scale_factor[1]))
        #self.gaussian2kp = gaussian2kp
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation3d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(device1)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}

        return kp

    def kp2heatmap(self, mean, spatial_size, kp_variance=0.06):
        """
        Transform a keypoint into gaussian like representation
        """
        #mean = kp['value']#.to(device0)

        coordinate_grid = make_coordinate_grid(spatial_size, mean.type()).to(device1)
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

        return out#.sum(dim=1).unsqueeze(1)

    def forward(self, x, scales_cus=True):
        #if scales_cus:
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)

        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)

        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)


        out = self.gaussian2kp(heatmap)
        out['predicted_map'] = heatmap
        out['heatmap'] = self.kp2heatmap(out['value'], x.shape[2:])
        

        return out
