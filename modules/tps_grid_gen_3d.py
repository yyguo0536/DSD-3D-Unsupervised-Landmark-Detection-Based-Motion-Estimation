# encoding: utf-8

import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
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

class TPSGridGen(nn.Module):

    def __init__(self, target_depth, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 3
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points#.float()

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

        # create target cordinate matrix
        HW = target_height * target_width * target_depth
        '''X = torch.arange(target_width).float()
        Y = torch.arange(target_height).float()
        Z = torch.arange(target_depth).float()

        X = (2 * (X / (target_width - 1)) - 1)
        Y = (2 * (Y / (target_height - 1)) - 1)
        Z = (2 * (Z / (target_depth - 1)) - 1)

        Z = Z.view(-1, 1, 1).repeat(1, target_height, target_width)
        Y = Y.view(1, -1, 1).repeat(target_depth, 1, target_width)
        X = X.view(1, 1, -1).repeat(target_depth, target_height, 1)

        Z = Z.view(1,-1)
        Y = Y.view(1,-1)
        X = X.view(1,-1)

        target_coordinate = torch.cat([Z, Y, X], dim = 1)'''

        target_coordinate = list(itertools.product(range(target_depth), range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate).to(device0) # HW x 2
        #print('2')
        Z, Y, X = target_coordinate.split(1, dim = 1)
        Z = Z * 2 / (target_depth - 1) - 1
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y, Z], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1).to(device0), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(4, 3).to(device0))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 3
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 4, 3))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate

if __name__ == '__main__':
    #main()
    target_p = torch.randn(6,3)
    source_p = torch.randn(1,6,3)
    tps = TPSGridGen(20,20,20,target_p)
    tmp = tps(source_p)
    print(tmp)