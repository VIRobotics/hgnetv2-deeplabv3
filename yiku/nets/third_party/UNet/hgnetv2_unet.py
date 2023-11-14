from yiku.PATH import WTS_STORAGE_DIR
from yiku.nets.BackBone.HGNetv2 import HG_backbone_l,HG_backbone_x
from torch import nn
import torch.nn.functional  as F
import torch
from yiku.utils.utils import fuse_conv_and_bn
class UP(nn.Module):
    def __init__(self, in_size, insize2,out_size,*args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size+insize2, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        try:
            x=self.up(inputs2)
            outputs = torch.cat([inputs1,x ], 1)
        except RuntimeError:
            print(inputs1.shape,x.shape)
            exit(1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class UNet(nn.Module):
    def __init__(self,  num_classes = 21, pretrained = False, **kwargs):
        super().__init__()
        self.bb = HG_backbone_l()
        if pretrained:
            self.bb.load_state_dict(torch.load(WTS_STORAGE_DIR/f"hgnetv2l.pt"))
        self.up_concat4 = UP(1024, 2048,out_size=512)# 16->32
        self.up_concat3 = UP(512,512,out_size=128)#32->64
        self.up_concat2 = UP(128, 128,out_size=128)
        self.final=nn.Conv2d(128, num_classes, 1)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def grad_backbone(self,need_grad=False):
        for param in self.bb.parameters():
            param.requires_grad = need_grad

    def forward(self,x):
        feature=[]
        for i,m in enumerate(self.bb.mlist):
            x=m(x)
            if i in [0,1,3,7,9]:
                feature.append(x)
        y=self.up_concat4(feature[3],feature[4])
        y=self.up_concat3(feature[2],y)
        y = self.up_concat2(feature[1], y)
        y=self.up_conv(y)
        y=self.final(y)
        y=F.interpolate(y,scale_factor=2,mode="bilinear")
        return y

    def fuse(self):
        self.bb=fuse_conv_and_bn(self.bb)
        return self