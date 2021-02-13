import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import itertools
import torch.nn as nn

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1-c)/(s**2))
    return rotation_matrix


def loss_center_dir_cos(x, y):
    ###########Measure the centering distance loss###########
    x_norm = torch.sqrt((x**2).sum(1))
    y_norm = torch.sqrt((y**2).sum(1))

    x_max = torch.max(x_norm)
    dist = torch.max(torch.zeros_like(y_norm), y_norm-x_max).max()
    
    return dist

def loss_centered_cos(x, y, num):
    #############lv and lvm distance distribution with their center###############

    x_center = torch.mean(x[-8:,:],dim=0).unsqueeze(0)
    y_center = torch.mean(y[-8:,:],dim=0).unsqueeze(0)

    x_centered = x - x_center
    y_centered = y - y_center

    x_dis = torch.sqrt((x_centered**2).sum(1))
    y_dis = torch.sqrt((x_centered**2).sum(1))

    x_max = torch.max(torch.sqrt((x_centered**2).sum(1)))
    y_max = torch.max(torch.sqrt((y_centered**2).sum(1)))

    x_norm = x_dis / x_max
    y_norm = y_dis / y_max

    remd = torch.dist(x_norm, y_norm)

    return remd

def compute_matrix(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 3) - control_points.view(1, M, 3)
    pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2)
    pairwise_dist = pairwise_similar_cos(pairwise_dist, pairwise_dist)
    # original implementation, very slow
    #pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    #pairwise_diff_square = pairwise_diff**2#(torch.sqrt((pairwise_diff**2)))/2#pairwise_diff * pairwise_diff
    #pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1] + pairwise_diff_square[:, :, 2]
    #pairwise_dist = torch.sqrt(pairwise_dist)#/3.5

    return pairwise_dist

def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 3) - control_points.view(1, M, 3)
    # original implementation, very slow
    pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    #pairwise_diff_square = pairwise_diff**2#(torch.sqrt((pairwise_diff**2)))/2#pairwise_diff * pairwise_diff
    #pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1] + pairwise_diff_square[:, :, 2]
    #pairwise_dist = torch.sqrt(pairwise_dist)#/3.5

    return pairwise_dist


def pairwise_distances_sq_l2(x, y):

    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def pairwise_similar_cos(x, y):

    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    
    dist = torch.mm(x, y_t)/x_norm/y_norm

    return dist

def pairwise_distances_cos(x, y):

    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    #dist = F.hardshrink(dist, lambd=0.036)

    return dist

def loss_distances_cos(x, y, num, lambd_value=0.025):

    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))

    dist_l2 = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist_l2 = torch.clamp(dist_l2, 1e-5, 1e5)/x.size(1)
    
    dist_cos = 1.-torch.mm(x, y_t)/x_norm/y_norm
    #dist_cos = F.hardshrink(dist_cos, lambd=0.036)
    #dist_cos = torch.clamp()
    #dist = F.softshrink(dist_cos, lambd=lambd_value) #+ 0.2 * dist_l2
    dist = dist_cos #+ dist_l2

    d_eye = torch.eye(num).to(device0)
    #CX_M_New = CX_M * (1.0-torch.eye(splits[0])).to(device0)
    dist_eye = dist * d_eye.float().to(device0)

    eye1 = torch.sum(dist_eye, 1)
    eye2 = torch.sum(dist_eye, 0)
    eye_m1,_ = dist.min(1)
    eye_m2,_ = dist.min(0)
    soft_eye_m1 = F.hardshrink(eye_m1, lambd=lambd_value)
    soft_eye_m2 = F.hardshrink(eye_m2, lambd=lambd_value)
    #soft_eye1 = F.hardshrink(eye1, lambd=lambd_value)
    #soft_eye2 = F.hardshrink(eye2, lambd=lambd_value)
    remd_mean = torch.max(eye_m1.mean(),eye_m2.mean())

    remd_eye = torch.max((eye1 - eye_m1).mean(),(eye2 - eye_m2).mean()) * 4.0 #+ remd_mean
    

    return remd_eye + remd_mean


def loss_adj(x, y, num, lambd_value=0.025):
    xx_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    xx_t = torch.transpose(x, 0, 1)
    dist_cos_x = torch.mm(x, xx_t)/xx_norm/xx_norm

    yy_norm = torch.sqrt((y**2).sum(1).view(-1, 1))
    yy_t = torch.transpose(y, 0, 1)
    dist_cos_y = torch.mm(y, yy_t)/yy_norm/yy_norm

    x_norm = torch.sqrt((dist_cos_x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(dist_cos_y, 0, 1)
    y_norm = torch.sqrt((dist_cos_y**2).sum(1).view(1, -1))

    dist_l2 = x_norm + y_norm - 2.0 * torch.mm(dist_cos_x, y_t)
    dist_l2 = torch.clamp(dist_l2, 1e-5, 1e5)/dist_cos_x.size(1)
    
    
    dist_cos = 1.-torch.mm(dist_cos_x, y_t)/x_norm/y_norm
    #dist_cos = F.hardshrink(dist_cos, lambd=0.036)
    #dist_cos = torch.clamp()
    #dist = F.softshrink(dist_cos, lambd=lambd_value) #+ 0.2 * dist_l2
    dist = dist_cos #+ dist_l2

    d_eye = torch.eye(num).to(device0)
    #CX_M_New = CX_M * (1.0-torch.eye(splits[0])).to(device0)
    dist_eye = dist * d_eye.float().to(device0)

    eye1 = torch.sum(dist_eye, 1)
    eye2 = torch.sum(dist_eye, 0)
    eye_m1,_ = dist.min(1)
    eye_m2,_ = dist.min(0)
    #soft_eye_m1 = F.hardshrink(eye_m1, lambd=lambd_value)
    #soft_eye_m2 = F.hardshrink(eye_m2, lambd=lambd_value)
    #soft_eye1 = F.hardshrink(eye1, lambd=lambd_value)
    #soft_eye2 = F.hardshrink(eye2, lambd=lambd_value)
    remd_mean = torch.max(eye_m1.mean(),eye_m2.mean())

    remd_eye = torch.max((eye1 - eye_m1).mean(),(eye2 - eye_m2).mean()) * 4.0 #+ remd_mean
    

    return remd_eye + remd_mean





def lr_distances(lvm, lv, rv):

    N = lvm.size(0)
    M = lv.size(0)
    K = rv.size(0)

    pairwise_diff_12 = (lvm.view(N, 1, 3) - lv.view(1, M, 3)) 
    pairwise_diff_13 = (lvm.view(N, 1, 3) - rv.view(1, K, 3)) 
    

    norm_12 = torch.sqrt((pairwise_diff_12**2).sum(-1))
    norm_13 = torch.sqrt((pairwise_diff_13**2).sum(-1))

    min_12, _ = norm_12.min(1)
    min_13, _ = norm_13.min(1)

    dis_loss = torch.max(torch.zeros_like(min_12),min_13 - min_12).max()
    
    return dis_loss

def get_DMat(X,Y,h=1.0,cb=0,splits=[128*3+256*3+512*4], cos_d=True):
    n = X.size(0)
    m = Y.size(0)
    M = Variable(torch.zeros(n,m)).to(device0)


    if 1:
        cb = 0
        ce = 0
        for i in range(len(splits)):
            if cos_d:
                ce = cb + splits[i]
                M = M + pairwise_distances_cos(X[:,cb:ce],Y[:,cb:ce])
            
                cb = ce
            else:
                ce = cb + splits[i]
                M = M + torch.sqrt(pairwise_distances_sq_l2(X[:,cb:ce],Y[:,cb:ce]))
            
                cb = ce

    return M




def viz_d(zx,coords):


    viz = zx[0][:,:1,:,:].clone()*0.

    for i in range(coords.shape[0]):
        vizt = zx[0][:,:1,:,:].clone()*0.

        for z in zx:
            cx = int(coords[i,0]*z.size(2))
            cy = int(coords[i,1]*z.size(3))

            anch = z[:,:,cx:cx+1,cy:cy+1]
            x_norm = torch.sqrt((z**2).sum(1,keepdim=True))
            y_norm = torch.sqrt((anch**2).sum(1,keepdim=True))
            dz = torch.sum(z*anch,1,keepdim=True)/x_norm/y_norm
            vizt = vizt+F.upsample(dz,(viz.size(2),viz.size(3)),mode='bilinear')*z.size(1)

        viz = torch.max(viz,vizt/torch.max(vizt))

    vis_o = viz.clone()
    viz = viz.data.cpu().numpy()[0,0,:,:]/len(zx)
    return vis_o



def remd_loss_vec(X,Y, h=None, cos_d=True, splits= [32],lambd_value=0.025):

    #X = compute_partial_repr(X, X)
    #Y = compute_partial_repr(Y, Y)
    N = X.size(0)
    M = Y.size(0)

    pairwise_diff_x = (X.view(N, 1, 3) - X.view(1, N, 3)) 
    '''+ \
        torch.eye(splits[0]).float().unsqueeze(-1).to(device0) * (X - x_center).repeat(splits[0],1,1)'''
    pairwise_diff_y = (Y.view(M, 1, 3) - Y.view(1, M, 3)) 
    '''+ \
        torch.eye(splits[0]).float().unsqueeze(-1).to(device0) * (Y - y_center).repeat(splits[0],1,1)'''

    d = X.size(0)

    CX_M_x = get_DMat(pairwise_diff_x[:,:,0],pairwise_diff_y[:,:,0],1.,cos_d=True, splits=splits)
    #CX_M_x = CX_M_x + get_DMat(pairwise_diff_x[:,:,0],pairwise_diff_y[:,:,0],1.,cos_d=False, splits=splits)
    CX_M_y = get_DMat(pairwise_diff_x[:,:,1],pairwise_diff_y[:,:,1],1.,cos_d=True, splits=splits)
    #CX_M_y = CX_M_y + get_DMat(pairwise_diff_x[:,:,1],pairwise_diff_y[:,:,1],1.,cos_d=False, splits=splits)
    CX_M_z = get_DMat(pairwise_diff_x[:,:,2],pairwise_diff_y[:,:,2],1.,cos_d=True, splits=splits)
    #CX_M_z = CX_M_z + get_DMat(pairwise_diff_x[:,:,2],pairwise_diff_y[:,:,2],1.,cos_d=False, splits=splits)
    #+get_DMat(X,Y,1.,cos_d=False, splits=splits)

    #CX_M_New = CX_M * (1.0-torch.eye(36)).to(device0)
    CX_M_EYE_x = CX_M_x * torch.eye(splits[0]).float().to(device0)
    CX_M_EYE_y = CX_M_y * torch.eye(splits[0]).float().to(device0)
    CX_M_EYE_z = CX_M_z * torch.eye(splits[0]).float().to(device0)

    tmp = torch.cat([CX_M_x.unsqueeze(-1), CX_M_y.unsqueeze(-1), CX_M_z.unsqueeze(-1)], dim=-1)
    #CX_M, _ = torch.max(tmp, dim=-1)
    #CX_M = torch.sqrt(CX_M_x**2+CX_M_y**2+CX_M_z**2)
    #CX_M = F.softshrink(CX_M, lambd=lambd_value)
    CX_M = CX_M_x + CX_M_y + CX_M_z

    #CX_M = torch.max(CX_M_x, CX_M_y, CX_M_z)
    CX_M_EYE = CX_M * torch.eye(splits[0]).float().to(device0)

    eye_m1,_ = CX_M.min(1)
    eye_m2,_ = CX_M.min(0)
    soft_eye_m1 = F.hardshrink(eye_m1, lambd = lambd_value)
    soft_eye_m2 = F.hardshrink(eye_m2, lambd = lambd_value)
    eye1 = torch.sum(CX_M_EYE, 1)
    eye2 = torch.sum(CX_M_EYE, 0)
    remd_mean = torch.max(eye_m1.mean(),eye_m2.mean())
    remd_eye = torch.max((eye1-eye_m1).mean(),(eye2-eye_m2).mean()) * 4.0

    eye1_x = torch.sum(CX_M_EYE_x, 1)
    eye2_x = torch.sum(CX_M_EYE_x, 0)
    eye_x_m1,_ = CX_M_x.min(1)
    eye_x_m2,_ = CX_M_x.min(0)

    eye1_y = torch.sum(CX_M_EYE_y, 1)
    eye2_y = torch.sum(CX_M_EYE_y, 0)
    eye_y_m1,_ = CX_M_y.min(1)
    eye_y_m2,_ = CX_M_y.min(0)

    eye1_z = torch.sum(CX_M_EYE_z, 1)
    eye2_z = torch.sum(CX_M_EYE_z, 0)
    eye_z_m1,_ = CX_M_z.min(1)
    eye_z_m2,_ = CX_M_z.min(0)

    remd_x = torch.max((eye1_x - eye_x_m1).mean(),(eye2_x - eye_x_m2).mean())
    remd_y = torch.max((eye1_y - eye_y_m1).mean(),(eye2_y - eye_y_m2).mean())
    remd_z = torch.max((eye1_z - eye_z_m1).mean(),(eye2_z - eye_z_m2).mean())
    remd = remd_x + remd_y + remd_z + remd_mean

    return remd_mean + remd_eye#_x + remd_y + remd_z



def remd_dis_vec(X,Y, h=None, cos_d=True, splits= [32],return_mat=False):

    #X = compute_partial_repr(X, X)
    #Y = compute_partial_repr(Y, Y)
    N = X.size(0)
    M = Y.size(0)

    pairwise_diff_x = (X.view(N, 1, 3) - X.view(1, N, 3)) 
    dist_x = torch.sqrt((pairwise_diff_x ** 2).sum(-1)+1e-5)**0.5
    norm_x = dist_x / dist_x.max()
    '''+ \
        torch.eye(splits[0]).float().unsqueeze(-1).to(device0) * (X - x_center).repeat(splits[0],1,1)'''
    pairwise_diff_y = (Y.view(M, 1, 3) - Y.view(1, M, 3)) 
    dist_y = torch.sqrt((pairwise_diff_y ** 2).sum(-1)+1e-5)**0.5
    norm_y = dist_y / dist_y.max()

    #x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    #y_t = torch.transpose(norm_y, 0, 1)
    #y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    
    #CX_M = 1.-torch.mm(norm_x, y_t)


    CX_M = get_DMat(norm_x,norm_y,1.,cos_d=True, splits=splits)
    
    #CX_M = CX_M + get_DMat(dist_x,dist_y,1.,cos_d=False, splits=splits)

    CX_M_EYE = CX_M * torch.eye(splits[0]).float().to(device0)

    eye_m1,_ = CX_M.min(1)
    eye_m2,_ = CX_M.min(0)
    eye1 = torch.sum(CX_M_EYE, 1)
    eye2 = torch.sum(CX_M_EYE, 0)
    remd_mean = torch.max(eye_m1.mean(),eye_m2.mean())
    remd_eye = torch.max((eye1-eye_m1).mean(),(eye2-eye_m2).mean()) #* 4.0


    return remd_mean + remd_eye


def remd_dis_loss(X,Y, h=None, cos_d=True, splits= [32],return_mat=False):

    N = X.size(0)
    M = Y.size(0)

    pairwise_diff_x = X.view(N, 1, 3) - X.view(1, N, 3)
    CX_M_x = pairwise_similar_cos(pairwise_diff_x[:,:,0],pairwise_diff_x[:,:,0])
    #CX_M_x = CX_M_x + get_DMat(pairwise_diff_x[:,:,0],pairwise_diff_y[:,:,0],1.,cos_d=False, splits=splits)
    CX_M_y = pairwise_similar_cos(pairwise_diff_x[:,:,1],pairwise_diff_x[:,:,1])
    #CX_M_y = CX_M_y + get_DMat(pairwise_diff_x[:,:,1],pairwise_diff_y[:,:,1],1.,cos_d=False, splits=splits)
    CX_M_z = pairwise_similar_cos(pairwise_diff_x[:,:,2],pairwise_diff_x[:,:,2])
    #CX_M_z = CX_M_z + get_DMat(pairwise_diff_x[:,:,2],pairwise_diff_y[:,:,2],1.,cos_d=False, splits=splits)
    '''+ \
        torch.eye(splits[0]).float().unsqueeze(-1).to(device0) * (X - x_center).repeat(splits[0],1,1)'''
    pairwise_diff_y = (Y.view(M, 1, 3) - Y.view(1, M, 3)) 
    CY_M_x = pairwise_similar_cos(pairwise_diff_y[:,:,0],pairwise_diff_y[:,:,0])
    #CX_M_x = CX_M_x + 0.2 * get_DMat(pairwise_diff_x[:,:,0],pairwise_diff_y[:,:,0],1.,cos_d=False, splits=splits)
    CY_M_y = pairwise_similar_cos(pairwise_diff_y[:,:,1],pairwise_diff_y[:,:,1])
    #CX_M_y = CX_M_y + 0.2 * get_DMat(pairwise_diff_x[:,:,1],pairwise_diff_y[:,:,1],1.,cos_d=False, splits=splits)
    CY_M_z = pairwise_similar_cos(pairwise_diff_y[:,:,2],pairwise_diff_y[:,:,2])

    C_M_x = get_DMat(CX_M_x,CY_M_x,1.,cos_d=True, splits=splits)
    #C_M_x = C_M_x + get_DMat(CX_M_x,CY_M_x,1.,cos_d=False, splits=splits)
    C_M_y = get_DMat(CX_M_y,CY_M_y,1.,cos_d=True, splits=splits)
    #C_M_y = C_M_y + get_DMat(CX_M_y,CY_M_y,1.,cos_d=False, splits=splits)
    C_M_z = get_DMat(CX_M_z,CY_M_z,1.,cos_d=True, splits=splits)
    #C_M_z = C_M_z + get_DMat(CX_M_z,CY_M_z,1.,cos_d=False, splits=splits)
    #+get_DMat(X,Y,1.,cos_d=False, splits=splits)

    #CX_M_New = CX_M * (1.0-torch.eye(36)).to(device0)
    C_M_EYE_x = C_M_x * torch.eye(splits[0]).float().to(device0)
    C_M_EYE_y = C_M_y * torch.eye(splits[0]).float().to(device0)
    C_M_EYE_z = C_M_z * torch.eye(splits[0]).float().to(device0)

    tmp = torch.cat([C_M_x.unsqueeze(-1), C_M_y.unsqueeze(-1), C_M_z.unsqueeze(-1)], dim=-1)
    #C_M, _ = torch.max(tmp, dim=-1)
    #C_M = torch.sqrt(C_M_x**2+C_M_y**2+C_M_z**2)
    C_M = C_M_x + C_M_y + C_M_z
    #CX_M = F.softshrink(CX_M, lambd=lambd_value)

    #CX_M = torch.max(CX_M_x, CX_M_y, CX_M_z)
    C_M_EYE = C_M * torch.eye(splits[0]).float().to(device0)

    eye_m1,_ = C_M.min(1)
    eye_m2,_ = C_M.min(0)
    eye1 = torch.sum(C_M_EYE, 1)
    eye2 = torch.sum(C_M_EYE, 0)
    remd_mean = torch.max(eye_m1.mean(),eye_m2.mean())
    remd_eye = torch.max((eye1-eye_m1).mean(),(eye2-eye_m2).mean()) * 4.0


    return remd_mean + remd_eye
    


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 3) - control_points.view(1, M, 3)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1] + pairwise_diff_square[:, :, 2]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    #print('finish')
    return repr_matrix

class AffGen(nn.Module):

    def __init__(self, target_control_points):
        super(AffGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 3
        N = target_control_points.size(0)
        self.num_points = N
        self.target_control_points = target_control_points.to(device0)#.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 4, N + 4).to(device0)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -4].fill_(1)
        forward_kernel[-4, :N].fill_(1)
        forward_kernel[:N, -3:].copy_(target_control_points)
        forward_kernel[-3:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        #print('2')

        

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(4, 3).to(device0))

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 3
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 4, 3))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        #source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        target_points = torch.cat(\
            [torch.ones(self.target_control_points.shape[0], 1).to(device0), \
            self.target_control_points], dim = 1)

        transformed_target = torch.matmul(Variable(target_points), mapping_matrix[:,-4:,:])

        return transformed_target
