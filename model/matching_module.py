import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
import sys
from misc.utils import *
from misc.layer import *
from torchsummary import summary
from model.ops.conv import BasicConv, BasicDeconv, ResBlock

from  spatial_correlation_sampler import SpatialCorrelationSampler

class CostVolume(nn.Module):
    def __init__(self, kernel_size, max_displacement, stride=1, abs_coordinate_output=False):
        super().__init__()
        self.correlation_layer = SpatialCorrelationSampler(kernel_size, 2*max_displacement + 1, stride,
                                                           int((kernel_size-1)/2))
        self.abs_coordinate_output = abs_coordinate_output

    def forward(self, feat1, feat2):
        assert feat1.dim() == 4 and feat2.dim() == 4, 'Expect 4 dimensional inputs'

        batch_size = feat1.shape[0]

        cost_volume = self.correlation_layer(feat1, feat2)

        # if self.abs_coordinate_output:
        #     cost_volume = cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])
        #     cost_volume = remap_cost_volume(cost_volume)

        return cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class Matching_Network(nn.Module):
    def __init__(self, displacement=10):
        super(Matching_Network, self).__init__()
        in_channels = [256,512,512]
        # self.de_pred = nn.Sequential(
        #     BasicConv(512, 256, kernel_size=3, padding=1, use_bn='bn'),
        #     BasicDeconv(256, 128, 2, stride=2, use_bn='bn'),
        #     BasicDeconv(128, 64, 2, stride=2, use_bn='bn'),
        #     BasicConv(64, 1, kernel_size=1, padding=0,  use_bn='none'),
        #     nn.Sigmoid()
        # )

        self.loc_layer = nn.Sequential(
            nn.Conv2d((displacement*2+1)**2, 512, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(768, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
            # # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            #
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # # nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            #
            # nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            # # nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            #
            # nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ReLU(inplace=True)
        )
        #
        # self.scale_layer = nn.Sequential(
        #     ResBlock(in_dim=768, out_dim=256, dilation=1, norm="bn"),
        #     ResBlock(in_dim=256, out_dim=128, dilation=2, norm="bn"),
        #     ResBlock(in_dim=128, out_dim=64, dilation=3, norm="bn"),
        #     ResBlock(in_dim=64, out_dim=32, dilation=4, norm="bn"),
        #     nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1, output_padding=0, bias=False),
        #     nn.ReLU(inplace=True)
        # )
        # # initialize_weights(self.loc_layer)
        # initialize_weights(self.scale_layer)

    def forward(self, x):
        # f = self.neck(f)
        # f =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2),F.interpolate(f[2],scale_factor=4)], dim=1)
        x = self.loc_layer(x)

        return x

class Matching_Network(nn.Module):
    def __init__(self, displacement=10):
        super(Matching_Network, self).__init__()
        in_channels = [256,512,512]
        # self.de_pred = nn.Sequential(
        #     BasicConv(512, 256, kernel_size=3, padding=1, use_bn='bn'),
        #     BasicDeconv(256, 128, 2, stride=2, use_bn='bn'),
        #     BasicDeconv(128, 64, 2, stride=2, use_bn='bn'),
        #     BasicConv(64, 1, kernel_size=1, padding=0,  use_bn='none'),
        #     nn.Sigmoid()
        # )

        self.loc_layer = nn.Sequential(
            nn.Conv2d(displacement**2, 512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # f = self.neck(f)
        # f =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2),F.interpolate(f[2],scale_factor=4)], dim=1)
        x = self.loc_layer(x)

        return x

def compute_cost_volume_wrong(features1, features2, max_displacement=4):
    """Compute the cost volume between features1 and features2.
    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.
    Args:
    features1: Tensor of shape [b, c, h, w]
    features2: Tensor of shape [b, c, h, w]
    max_displacement: int, maximum displacement for cost volume computation.
    Returns:
    tf.tensor of shape [b, (2 * max_displacement + 1) ** 2, h, w] of costs for
    all displacements.
    """
    _, _, height, width = features1.size()
    #     if max_displacement <= 0 or max_displacement >= height:
    #         raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed 
    # to compute the cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = F.pad(features2, (max_disp, max_disp, max_disp, max_disp))
    cost_list = []
    import pdb
    pdb.set_trace()
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(features1 * features2_padded[:, :, i:(height + i), j:(width + j)], dim=1, keepdim=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, dim=1)
    return cost_volume

def compute_cost_volume(features1, features2, max_displacement=4):
    """Compute the cost volume between features1 and features2.
    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.
    Args:
    features1: Tensor of shape [b, c, h, w]
    features2: Tensor of shape [b, c, h, w]
    max_displacement: int, maximum displacement for cost volume computation.
    Returns:
    tf.tensor of shape [b, (2 * max_displacement + 1) ** 2, h, w] of costs for
    all displacements.
    """
    N, C, height, width = features1.size()
    #     if max_displacement <= 0 or max_displacement >= height:
    #         raise ValueError(f'Max displacement of {max_displacement} is too large.')

    windows = max_displacement


    # Pad features2 and shift it while keeping features1 fixed
    # to compute the cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    # features2_padded = F.pad(features2, (max_disp, max_disp, max_disp, max_disp))
    # cost_list = []

    # for i in range(num_shifts):
    #     for j in range(num_shifts):
    #         corr = torch.mean(features1 * features2_padded[:, :, i:(height + i), j:(width + j)], dim=1, keepdim=True)
    #         cost_list.append(corr)
    # cost_volume = torch.cat(cost_list, dim=1)
    # import pdb

    print('start_processing')
    un = torch.nn.Unfold(kernel_size=windows, dilation=1, padding=windows//2, stride=1)
    s_b = un(features2).transpose(2, 1)
    B, N, S = s_b.size()
    # pdb.set_trace()
    s_b = s_b.view(B,N,C,-1)
    s_a = features1.view(B,C,-1).transpose(2,1).unsqueeze(3)
    score = torch.einsum('bsdn,bsdm->bsnm', s_a, s_b)  # [1,1,225]
    # for i in range(height):
    #     for j in range(width):
    #         # corr = torch.mean(features1 * features2_padded[:, :, i:(height + i), j:(width + j)], dim=1, keepdim=True)
    #         s_a = features1[:,:,i,j].unsqueeze(2) # [1,256,1]
    #         s_b = s_b[:, :, i:(num_shifts + i), j:(num_shifts + j)] #[1,256,15,15]
    #         s_b = torch.flatten(s_b,start_dim=2) #[1,256,15**2]
    #
    #         cost_list.append(score)

    # print('feature_similarity', scores, scores.size())
    # cost_volume = torch.cnat(cost_list, dim=1)
    score = score.squeeze(2).transpose(2,1).view(B,windows**2,height,width)/256.
    return score

def normalize_features(feature_list, normalize=True, center=True, moments_across_channels=True,
                       moments_across_images=True):
    """Normalizes feature tensors (e.g., before computing the cost volume).
    Args:
        feature_list: list of Tensors, each with dimensions [b, c, h, w]
        normalize: bool flag, divide features by their standard deviation
        center: bool flag, subtract feature mean
        moments_across_channels: bool flag, compute mean and std across channels
        moments_across_images: bool flag, compute mean and std across images
    Returns:
        list, normalized feature_list
    """

    # Compute feature statistics

    statistics = collections.defaultdict(list)
    dim = (1, 2, 3) if moments_across_channels else (2, 3)
    for feature_image in feature_list:
        variance, mean = torch.var_mean(feature_image, dim=dim, keepdim=True, unbiased=False)
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if moments_across_images:
        statistics['mean'] = [torch.mean(torch.stack(statistics['mean']))] * len(feature_list)
        statistics['var'] = [torch.mean(torch.stack(statistics['var']))] * len(feature_list)

    statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

    # Center and normalize features
    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ]

    if normalize:
        feature_list = [
            f / std for f, std in zip(feature_list, statistics['std'])
        ]
    return feature_list