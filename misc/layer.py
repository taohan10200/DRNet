import torch
import torch.nn as nn

from .dot_ops import Gaussian, SumPool2d
import scipy.spatial
import scipy.ndimage
import numpy as np
import  torch.nn.functional as F
import cv2 as cv
class Point2Mask(object):
    def __init__(self,  max_kernel_size=7):

        self.max_kernel_size = max_kernel_size
    def __call__(self, target, pre_map):
        b,c,h,w = pre_map.size()
        mask_map = torch.zeros_like(pre_map)
        for idx, sub_target in enumerate(target):
            points = sub_target["points"]
            # import pdb
            # pdb.set_trace()
            count = points.shape[0]
            if count==0:
                continue
            elif count==1:
                pt = points[0].astype(np.int32)
                kernel_size = self.max_kernel_size
                up = max(pt[1] - kernel_size, 0)
                down = min(pt[1] + kernel_size + 1, h)
                left = max(pt[0] - kernel_size, 0)
                right = min(pt[0] + kernel_size + 1, w)

                mask_map[idx, 0, up:down + 1, left:right + 1] = 1
            else:
                leafsize = 2048
                tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
                distances, locations = tree.query(points, k=2)
                for i, pt in enumerate(points):
                    if pt[0] >= w or pt[1] > h:
                        continue
                    pt = pt.astype(np.int32)
                    kernel_size = (distances[i][1]) * 0.25
                    kernel_size = min(self.max_kernel_size, int(kernel_size + 0.5))
                    up = max(pt[1] - kernel_size,0)
                    down = min(pt[1] + kernel_size+1,h)
                    left = max(pt[0] - kernel_size,0)
                    right = min(pt[0] + kernel_size+1,w)
                    mask_map[idx,0, up:down+1, left:right+1]=1

                # density_nn[np.where(pnt_density > 0)] = distances[i][1]
                # mask_map += pnt_density
                # density_std[np.where(pnt_density > 0)] = sigma
        # mask_map = mask_map.astype(np.uint8) * 255
        # cv.imwrite('../dataset/mask_vis/mask_vis.png', mask_map[0][0].cpu().numpy(), [cv.IMWRITE_PNG_BILEVEL, 1])
        # import pdb
        # pdb.set_trace()
        # print(mask_map.sum())
        return  mask_map
class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=15):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size//2, froze=True)
    
    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(9,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)


        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]


        fea = torch.cat(fea_stack, 3)
        return fea