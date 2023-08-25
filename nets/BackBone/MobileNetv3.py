from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights,mobilenet_v3_large,MobileNet_V3_Large_Weights
import sys
from torch import nn
class MNv3_backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        pretrained = bool(kwargs.get("pretrained",True))
        self.low_ch=24
        if "arch" not in kwargs.keys() :
            arch = "small"
        else :
            arch = "large" if "s" not in  str(kwargs["arch"]) else "small"
        if arch=="large":
            self.feature_ch = 960
        else:
            self.feature_ch = 576
        for key in list(kwargs.keys()):
            kwargs.__delitem__(key)
        super().__init__(*args, **kwargs)
        mod = sys.modules[__name__]
        func=getattr(mod, f"mobilenet_v3_{arch}")
        wts=getattr(mod, f"MobileNet_V3_{arch.capitalize()}_Weights")
        if pretrained:
            model = func(weights=wts.IMAGENET1K_V1)
        else:
            model = func()
        self.__model = model.features

    def forward(self, x):
        for i, n in enumerate(self.__model):
            x = n(x)
            if i == 3:
                low = x
        return low, x

def mobilenetv3l(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="large")

def mobilenetv3s(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="small")