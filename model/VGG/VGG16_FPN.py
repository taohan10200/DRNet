from  torchvision import models
import sys
import torch.nn.functional as F
from misc.utils import *
from misc.layer import *
from torchsummary import summary
from model.necks import FPN
from .conv import ResBlock

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class VGG16_FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FPN, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())

        self.layer1 = nn.Sequential(*features[0:23])
        self.layer2 = nn.Sequential(*features[23:33])
        self.layer3 = nn.Sequential(*features[33:43])

        in_channels = [256,512,512]
        self.neck = FPN(in_channels,192,len(in_channels))
        self.neck2f = FPN(in_channels, 128, len(in_channels))
        self.loc_head = nn.Sequential(
            nn.Dropout2d(0.2),
            ResBlock(in_dim=576, out_dim=256, dilation=0, norm="bn"),
            ResBlock(in_dim=256, out_dim=128, dilation=0, norm="bn"),

            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.feature_head = nn.Sequential(
            nn.Dropout2d(0.2),
            ResBlock(in_dim=384, out_dim=384, dilation=0, norm="bn"),
            ResBlock(in_dim=384, out_dim=256, dilation=0, norm="bn"),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        f_list = []
        x = self.layer1(x)
        f_list.append(x)
        x2 = self.layer2(x)
        f_list.append(x2)
        x = self.layer3(x2)
        f_list.append(x)


        f = self.neck(f_list)
        f =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)

        x = self.loc_head(f)

        f = self.neck2f(f_list)
        f =torch.cat([f[0],  F.interpolate(f[1],scale_factor=2,mode='bilinear',align_corners=True),
                      F.interpolate(f[2],scale_factor=4, mode='bilinear',align_corners=True)], dim=1)
        feature = self.feature_head(f)
        return feature, x



