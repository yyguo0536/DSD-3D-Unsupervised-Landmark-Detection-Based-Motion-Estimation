from torch import nn
import torch.nn.functional as F
import torch
from modules.util_3d import Hourglass_D, AntiAliasInterpolation3d, make_coordinate_grid, kp2gaussian
import fractions
from modules.tps_grid_gen_3d import *

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, motion_num_kp, still_num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass_D(in_chal=(motion_num_kp + 1) * (num_channels + 1),
                                   out_chal=motion_num_kp + 1)

        self.mask = nn.Conv3d(motion_num_kp + 1, motion_num_kp + 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        

        if estimate_occlusion_map:
            self.occlusion = nn.Conv3d(num_blocks, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        else:
            self.occlusion = None

        self.num_kp = motion_num_kp
        self.still_num_kp = still_num_kp
        self.scale_factor = float(fractions.Fraction(scale_factor[0],scale_factor[1]))
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation3d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, \
            kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, \
            kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type()).to(device0)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        #heatmap = heatmap.unsqueeze(2)
        return heatmap, gaussian_source

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, d, h, w = source_image.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type()).to(device0)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)


        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions, identity_grid[:,0,:,:,:,:]

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        """
        bs, _, d, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), d, h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, d, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, d, h, w = source_image.shape

        kp_source_motion = {'value': kp_source['value'][:,self.still_num_kp:,:].to(device0)}
        kp_driving_motion = {'value': kp_driving['value'][:,self.still_num_kp:,:].to(device0)}

        out_dict = dict()
        heatmap_representation, heatmap_source = self.create_heatmap_representations(source_image, kp_driving_motion, kp_source_motion)
        sparse_motion, ref_grid = self.create_sparse_motions(source_image, kp_driving_motion, kp_source_motion)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source
        out_dict['source_heatmap'] = heatmap_source.sum(dim=1).unsqueeze(1)

        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, d, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)
        deformation = ((sparse_motion * mask).sum(dim=1)).permute(0,2,3,4,1)

        out_dict['deformation'] = deformation


        return out_dict
