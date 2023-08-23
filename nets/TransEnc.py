from ultralytics.nn.modules import AIFI,Conv
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransEnc(nn.Module):
    def __init__(self, *args,dim_in=2048,dim_out=256,bn_mom=0.1,rate=1,**kwargs):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.__AIFI=AIFI(dim_in,1024,8)
        self.__conv1x1=Conv(dim_in, 256*3, 1, 1, None, 1, 1, False)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        x0=self.branch1(x)
        x1=self.__AIFI(x)
        x1=self.__conv1x1(x1)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([x0,x1, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

