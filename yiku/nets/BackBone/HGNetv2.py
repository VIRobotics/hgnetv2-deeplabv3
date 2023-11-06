import torch
from torch import nn
import sys
from yiku.PATH import WTS_STORAGE_DIR
from nets.modules.HGBlock import HGStem,HGBlock
from nets.modules.block import DWConv


class HG_backbone_x(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlist = nn.ModuleList(
            [HGStem(3, 32, 64),
             HGBlock(64, 64, 128, 3, n=6),

             DWConv(128, 128, 3, 2, 1, False),
             HGBlock(128, 128, 512, 3, n=6),
             HGBlock(512, 128, 512, 3, n=6,lightconv=False,shortcut=True),

             DWConv(512, 512, 3, 2, 1, False),
             HGBlock(512, 256, 1024, 5, lightconv=True, shortcut=False, n=6),
             HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
             HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
             HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),
             HGBlock(1024, 256, 1024, 5, lightconv=True, shortcut=True, n=6),

             DWConv(1024, 1024, 3, 2, 1, False),
             HGBlock(1024, 512, 2048, 5, lightconv=True, shortcut=False, n=6),
             HGBlock(2048, 512, 2048, 5, lightconv=True, shortcut=True, n=6)]
        )

    def forward(self, x):
        for modules in self.mlist:
            x = modules(x)
        return x
class HG_backbone_l(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlist=nn.ModuleList(
            [HGStem(3, 32, 48),
            HGBlock(48, 48, 128, 3, n=6),

            DWConv(128, 128, 3, 2, 1, False),
            HGBlock(128, 96, 512, 3, n=6),

            DWConv(512, 512, 3, 2, 1, False),
            HGBlock(512, 192, 1024, 5,lightconv=True,shortcut=False,n=6),
            HGBlock(1024, 192, 1024, 5, lightconv=True, shortcut=True, n=6),
            HGBlock(1024, 192, 1024, 5, lightconv=True, shortcut=True, n=6),

            DWConv(1024, 1024, 3, 2, 1, False),
            HGBlock(1024, 384, 2048, 5, lightconv=True, shortcut=False, n=6)]
        )

    def forward(self, x):
        for modules in self.mlist:
            x=modules(x)
        return x

class HG_backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        pretrained =bool(kwargs.get("pretrained",True))
        arch =str(kwargs.get("arch","l"))[0]
        for key in list(kwargs.keys()):
            kwargs.__delitem__(key)
        super().__init__(*args, **kwargs)
        mod = sys.modules[__name__]
        func = getattr(mod, f"HG_backbone_{arch}")
        m=func()
        if pretrained:
            m.load_state_dict(torch.load(WTS_STORAGE_DIR/f"hgnetv2{arch}.pt"))
        self.__model = m
        self.__dict__.update(**{"l":{"feature_ch":2048,"low_ch":512},
                                "x":{"feature_ch":2048,"low_ch":512}
                                }[arch])

    def forward(self, x):
        for i, n in enumerate(self.__model.mlist):
            x = n(x)
            if i == 9:# Get the featuremap
                break
            if i == 3:
                low = x # low feature

        return low, x

def hgnetv2l(pretrained=True, **kwargs):
    return HG_backbone(pretrained=pretrained,arch="l")

def hgnetv2x(pretrained=True, **kwargs):
    return HG_backbone(pretrained=pretrained,arch="x")

if __name__=="__main__":
    class M(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.m=HG_backbone_l()

        def forward(self,x):
            for i,modules in enumerate(self.m.mlist):
                x = modules(x)
                print(i,x.shape)
            return x
    m=M()
    x=torch.rand(1,3,512,512)
    m(x)
