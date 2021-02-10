import torch
from torch import nn
import torch.nn.functional as F
from modules.util_3d import ResBlock3d, SameBlock3d, UpBlock2d, DownBlock2d
from modules.dense_motion_3d import DenseMotionNetwork
from modules.dmotion_3d import DenseMotionNetwork
from modules.dmotion_3d_new import DenseMotionNetworkNew
from modules.vnet_3d import Temporal_Encoder, Temporal_Decoder, Temporal_Decoder_G, Temporal_Encoder_G
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
class OcclusionAwareGenerator(nn.Module):
    """
    """

    def __init__(self, num_channels, motion_num_kp, still_num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(motion_num_kp=motion_num_kp, still_num_kp = still_num_kp,
                                                            num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None


        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if h_old != h or w_old != w or d_old != d:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            deformation = dense_motion['deformation']
            out_deformed = []

            output_dict["deformed"] = self.deform_input(source_image, deformation)
            output_dict["deformation"] = deformation
        out = source_image

        output_dict["prediction"] = out

        return output_dict
