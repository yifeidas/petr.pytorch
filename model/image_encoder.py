import torch
from torch import nn
import torch.nn.functional as F
from model.network_blocks import BaseConv
from model.darknet import CSPDarknet
import os

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class CamEncoder(nn.Module):
    def __init__(self, C, backbone='yolo-m'):
        super(CamEncoder, self).__init__()
        self.C = C

        if backbone == 'yolo-m':
            self.trunk = CSPDarknet(0.67, 0.75, act='relu')
            pth = 'pretrain/CSPDarknet.pth'
            self.up = Up(768+384, C)
        else:
            self.trunk = CSPDarknet(1, 1, stem='focus', act='silu')
            pth = 'pretrain/CSPDarknet-l.pth'
            self.up = Up(1024+512, C)
        
        if os.path.exists(pth):
            print('load backbone %s'%pth)
            checkpoint = torch.load(pth, map_location=torch.device('cpu'))
            self.trunk.load_state_dict(checkpoint['net'])
        
        
    def forward(self, x):
        # return B*N x C x H x W
        x_dict = self.trunk(x)
        x = self.up(x_dict['dark5'], x_dict['dark4'])

        return x