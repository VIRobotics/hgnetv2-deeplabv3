import torch
import torch.nn as nn
import torch.nn.functional as F

from yiku.nets.modules.feature_decoder import TransEnc,ASPP
from yiku.nets.Head.BaseLab import BaseLabHeader

class LabHeader(BaseLabHeader):
    def __init__(self, H, W, num_classes: int, low_f_ch: int, f_ch: int, downsample_factor=16, enc="ASPP",*args, **kwargs):
        super().__init__(H, W, num_classes, low_f_ch, f_ch, downsample_factor, *args, **kwargs)
        if enc=="transformer":
            self.aspp=TransEnc(dim_in=f_ch, dim_out=256, rate=16 // downsample_factor)
        else:
            self.aspp = ASPP(dim_in=f_ch, dim_out=256, rate=16 // downsample_factor)

    def forward(self, low_level_features, features):
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        x = features
        # print(low_level_features.shape, x.shape)
        x = self.aspp(x)
        return self.output(low_level_features,x)


def aspp(H, W, num_classes, low_level_channels, in_channels,**kwargs):
    return LabHeader(H, W, num_classes, low_level_channels, in_channels,enc="ASPP",**kwargs)

def transformer(H, W, num_classes, low_level_channels, in_channels,**kwargs):
    return LabHeader(H, W, num_classes, low_level_channels, in_channels,enc="transformer",**kwargs)