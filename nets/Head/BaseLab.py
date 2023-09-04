import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.modules.feature_decoder import TransEnc
from nets.modules.block import C2f,C2TR

class BaseLabHeader(nn.Module):
    def __init__(self, H, W, num_classes: int, low_f_ch: int, f_ch: int, downsample_factor=16, *args,
                 **kwargs):
        super().__init__()
        use_c2f=kwargs.get("use_c2f",False)
        self.size = (H, W)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_f_ch, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        if use_c2f:
            self.cat_conv = nn.Sequential(
                C2f(48 + 256, 256, shortcut=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.5),

                C2TR(256, 256, shortcut=True),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
            )
        else:
            self.cat_conv = nn.Sequential(
                nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Dropout(0.1),
            )

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)


    def output(self,low_level_features,x):
        H, W = self.size
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

