from torch import nn
import torch
import torch.nn.functional as F
from modules.util_3d import AntiAliasInterpolation3d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
from modules.vnet_3d import Temporal_Encoder, Temporal_Decoder, Temporal_VGG
from .pyr_lap import dec_lap_pyr, syn_lap_pyr, syn_lap_pyr_edge
from gcn_layer import GCN_Seg
import pickle
from emd import EMDLoss
from modules.ot_loss import remd_dis_loss, remd_loss_vec, remd_dis_vec, loss_centered_cos, pairwise_distances_cos, loss_distances_cos
from modules.tps_grid_gen_3d import *


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 3) - control_points.view(1, M, 3)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1] + pairwise_diff_square[:, :, 2]
    #repr_matrix = torch.sqrt(pairwise_diff)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    #mask = repr_matrix != repr_matrix
    #repr_matrix.masked_fill_(mask, 0)
    #print('finish')
    return pairwise_dist


def create_adj(edge):
    shape = len(edge)
    adj = torch.zeros(shape, shape)
    for i in range(shape):
        for j in range(len(edge[i])):
            tmp_x = int(edge[i][j][0])#torch.Tensor([edge[i][j][0]]).type(torch.IntTensor)
            tmp_y = int(edge[i][j][1])#torch.Tensor([edge[i][j][1]]).type(torch.IntTensor)
            adj[tmp_x, tmp_y] = 1.0
    return adj




def deform_heatmap(inp, deformation):
    _, d_old, h_old, w_old, _ = deformation.shape
    _, _, d, h, w = inp.shape
    if h_old != h or w_old != w or d_old != d:
        deformation = deformation.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
        deformation = deformation.permute(0, 2, 3, 4, 1)
    return F.grid_sample(inp, deformation)


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.endocer = Temporal_Encoder(1)

    def forward(self, X):
        self.endocer = Temporal_Encoder(X)
        return out

def gaussian2kp(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(device0)
    value = (heatmap * grid).sum(dim=(2, 3, 4))
    

    return value

def kp2heatmap(mean, spatial_size, kp_variance=0.01):
    """
    Transform a keypoint into gaussian like representation
    """
    #mean = kp['value']#.to(device0)

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

    return out#.sum(dim=1).unsqueeze(1)


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation3d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 3, 4]))#.to(device1)#.cuda()
        #noise[:,:,2] = 0
        #noise[:,2,:2] = 0
        #theta = torch.eye(3,4).view(1,3,4)
        self.theta = torch.eye(3, 4).view(1, 3, 4)#.to(device1)#.cuda() noise +
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps'], kwargs['points_tps']), type=noise.type())#.to(device1)#.cuda()
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 3]))#.to(device1)#.cuda()
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)#.to(device1)#.cuda()
        grid = grid.view(1, frame.shape[2] * frame.shape[3] * frame.shape[4], 3)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], frame.shape[4], 3)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :3], coordinates.unsqueeze(-1)) + theta[:, :, :, 3:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 3) - control_points.view(1, 1, -1, 3)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        grad_z = grad(new_coordinates[..., 2].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2), grad_z[0].unsqueeze(-2)], dim=-2)
        return jacobian



def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, common_params, dict_data):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        self.disc_pyramid = ImagePyramide(self.disc_scales, 1)
        self.emd_dist =  EMDLoss()
        self.loss_L2 = torch.nn.MSELoss()
        self.loss_L1 = torch.nn.L1Loss()
        self.maxPool = torch.nn.MaxPool3d(5,1,2).cuda()
        self.motion_kp_num = common_params['motion_num_kp']
        self.still_kp_num = common_params['still_num_kp']

        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
            self.disc_pyramid = self.disc_pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.kp2gaussian_func = kp2heatmap

        self.appearance = syn_lap_pyr_edge
        self.dec_lap_pyr = dec_lap_pyr#.cuda() 
        self.syn_lap_pyr = syn_lap_pyr#.cuda()
        

        self.vertex = torch.from_numpy(np.array(dict_data['vertext']).astype(np.float32)).to(device0)#.unsqueeze(0)
        edge = np.array(dict_data['edge'])
        adj = create_adj(edge).cuda()

        adj = adj + adj.t()

        e, v = torch.symeig(adj, eigenvectors=True)

        self.adj = torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2,-1)))

        if sum(self.loss_weights['perceptual']) != 0:
            self.dec_lap_pyr = dec_lap_pyr#.cuda() 
            self.syn_lap_pyr = syn_lap_pyr#.cuda()

    def forward(self, x, aff=False):


        kp_source = self.kp_extractor((x['source']).to(device1))
        kp_driving = self.kp_extractor((x['driving']).to(device1))

        transform_ref = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
        vertex_transformed = transform_ref.warp_coordinates(self.vertex.unsqueeze(0))[0,:,:]


        kp_source['predicted_map'] = kp_source['predicted_map'].to(device0)
        kp_driving['predicted_map'] = kp_driving['predicted_map'].to(device0)
        

        
        kp_source['value'] = kp_source['value'].to(device0)
        kp_driving['value'] = kp_driving['value'].to(device0)


        generated = self.generator(x['source'].to(device0), kp_source=kp_source, kp_driving=kp_driving)

        bs, _, d, h, w = x['source'].shape

        '''tps = TPSGridGen(96, 96, 96, inter_kp[0,:,:]).to(device0)
        tps_source_grid = tps(kp_source['value'])
        tps_source_grid = tps_source_grid.view(1, 96, 96, 96, 3)

        tps_driving_grid = tps(kp_driving['value'])
        tps_driving_grid = tps_driving_grid.view(1, 96, 96, 96, 3)

        deformed_tps_source = deform_heatmap(x['source'].to(device0), tps_source_grid)
        deformed_tps_driving = deform_heatmap(x['driving'].to(device0), tps_driving_grid)'''

        #deformed_source_grad = deform_heatmap(source_grad_array,generated['deformation'])

        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'].to(device0))
        pyramide_deformed = self.pyramid(generated['deformed'].to(device0))

        source_heatmap_gaussian = self.kp2gaussian_func(kp_source['value'].to(device0), \
            (x['source'].shape[2]//2, x['source'].shape[3]//2, x['source'].shape[4]//2), 0.001)

        source_heatmap_combined = F.interpolate(source_heatmap_gaussian, \
            size=(x['source'].shape[2],x['source'].shape[3],x['source'].shape[4]), mode='trilinear').sum(1).unsqueeze(1)

        driving_heatmap_gaussian = self.kp2gaussian_func(kp_driving['value'].to(device0), \
            (x['source'].shape[2]//2, x['source'].shape[3]//2, x['source'].shape[4]//2), 0.001)

        
        kp_bias = source_heatmap_gaussian * driving_heatmap_gaussian

        kp_bias = kp_bias.view(kp_bias.shape[0], kp_bias.shape[1], -1)
        kp_bias = kp_bias.sum(dim=-1) / x['deformed_weights'].to(device0)

        loss_bias = kp_bias.sum()


        weights_kp = kp_source['heatmap'].sum(dim=1).max().to(device0)
        loss_heatmap = F.relu(kp_source['heatmap'].sum(dim=1).unsqueeze(1)-1).sum() / \
            (self.still_kp_num+self.motion_kp_num)
        loss_heatmap = loss_heatmap.to(device0)

        

        loss_values['kp_sum_value'] = loss_heatmap  * \
            self.loss_weights['kp_sum_value'] / x['source_value'].to(device0) 

        loss_values['bias'] = loss_bias * self.loss_weights['union']
        
        inter_kp = (kp_driving['value']+kp_source['value']) / 2.0

        vertex_transformed = vertex_transformed.unsqueeze(0).to(device0)

        loss_values['loss_dis'] = remd_dis_loss(inter_kp[0,:,:].to(device0), \
                    vertex_transformed[0,:,:].to(device0), splits=[self.rv_num+self.lvm_num+self.lv_num]) * 5.0
        

        #loss_values['dis'] = (loss_distances_cos(inter_kp[0,:,:].to(device0) - inter_kp[0,:,:].mean(0).to(device0), \
                    #vertex_transformed[0,:,:].to(device0) - vertex_transformed[0,:,:].mean(0).to(device0), self.rv_num)) * 10.0
        
        loss_remd_all = remd_loss_vec(inter_kp[0,:,:], vertex_transformed[0,:,:], \
            splits=[self.still_kp_num+self.motion_kp_num], lambd_value=0.036)

        loss_values['remd_dis'] = loss_remd_all * 20.0

        


        deformed_source_kp = deform_heatmap(kp_source['predicted_map'], generated['deformation'].to(device0))
        deformed_source_flatten = deformed_source_kp.view(deformed_source_kp.shape[0], deformed_source_kp.shape[1], -1)
        
        deformed_sum = deformed_source_flatten.sum(dim=-1).unsqueeze(-1)
        deformed_source_norm = deformed_source_flatten / (deformed_sum + 1e-6)
        deformed_source_norm = deformed_source_norm.view(deformed_source_kp.shape)
        deformed_source_value = gaussian2kp(deformed_source_norm)
        #loss_deformed_hm = self.loss_L2(driving_heatmap, deformed_source_value)
        loss_deformed_hm = torch.abs(kp_driving['value'] - deformed_source_value).mean()
        #loss_deformed_hm = torch.abs(deformed_source_kp-driving_heatmap).sum(dim=(2,3,4)).mean()
        loss_values['deformed_hm'] = loss_deformed_hm * 100.0 #* x['deformed_weights'].to(device0)

        ref_hp = self.kp2gaussian_func(vertex_transformed.to(device0), (80,80,80), 0.006)
        ref_hp = F.interpolate(ref_hp, \
            size=(160,160,160), mode='trilinear')#.sum(1).unsqueeze(1)
        generated.update({'ref_hp': ref_hp})


        if sum(self.loss_weights['perceptual']) != 0:
            value_total1 = 0
            value_total2 = 0
            value_total3 = 0
            for i, scale in enumerate(self.scales):

                self.loss_L2 = self.loss_L2.to(device0)
                tmp_deformed = pyramide_deformed['prediction_' + str(scale)]
                tmp_target = pyramide_real['prediction_' + str(scale)]
                value1 = self.loss_L2(tmp_deformed, tmp_target)
                value_total1 += self.loss_weights['deformed'][i] * (value1) 
                loss_values['deformed'] = value_total1


        

        

        if self.loss_weights['generator_gan'] != 0:
            dv_kp = self.kp2gaussian_func(kp_driving['value'].to(device0), (80,80,80), 0.004)
            dv_kp = torch.sum(dv_kp, dim=1).unsqueeze(1)
            _, discriminator_maps_generated = self.discriminator(dv_kp)
            value_total = 0
            value = ((1 - discriminator_maps_generated) ** 2).mean()
            loss_values['gen_gan'] = self.loss_weights['generator_gan'] * value

            '''if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total'''

        if self.loss_weights['equivariance_value'] != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving']).to(device1)
            transformed_kp = self.kp_extractor(transformed_frame)


            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            value = torch.abs(kp_driving['value'].to(device0) - transform.warp_coordinates(transformed_kp['value'].to(device0))).mean()
                
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value 
                

            
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, dict_data):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        

        self.vertex = torch.from_numpy(np.array(dict_data['vertext']).astype(np.float32)).to(device0)

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):

        transform_ref = Transform(1, **self.train_params['transform_params'])
        vertex_transformed = transform_ref.warp_coordinates(self.vertex.unsqueeze(0))#.permute(0,2,1)

        x_kp = kp2heatmap(vertex_transformed, (80,80,80), 0.004)
        x_kp = torch.sum(x_kp, dim=1).unsqueeze(1)
        generated_kp = kp2heatmap(detach_kp(generated['kp_driving'])['value'], (80,80,80), 0.004)
        generated_kp = torch.sum(generated_kp, dim=1).unsqueeze(1)
        #pyramide_real = self.pyramid(x['driving'].to(device0))
        #pyramide_generated = self.pyramid(generated['deformed'].detach())

        #kp_driving = generated['kp_driving']['value']
        _, discriminator_maps_generated = self.discriminator(generated_kp)
        _, discriminator_maps_real = self.discriminator(x_kp)
        value = (1 - discriminator_maps_real) ** 2 + discriminator_maps_generated ** 2

        loss_values = {}
        '''value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()'''
        loss_values['disc_gan'] = self.loss_weights['discriminator_gan'] * value.mean()

        return loss_values




