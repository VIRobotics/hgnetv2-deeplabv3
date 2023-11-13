import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights,mobilenet_v3_large,MobileNet_V3_Large_Weights
import sys
from yiku.PATH import WTS_STORAGE_DIR
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
            self.low_ch = 40
            self._i = 6
            wts = WTS_STORAGE_DIR / "mobilenet_v3_large-5c1a4163.pth"
        else:
            self._i = 3
            self.low_ch = 24
            self.feature_ch = 576
            wts=WTS_STORAGE_DIR/"mobilenet_v3_small-047dcff4.pth"
        for key in list(kwargs.keys()):
            kwargs.__delitem__(key)
        super().__init__(*args, **kwargs)
        mod = sys.modules[__name__]
        func=getattr(mod, f"mobilenet_v3_{arch}")
        if pretrained:
            model = func()
            model.load_state_dict(torch.load(wts))
        else:
            model = func()
        self.__model = model.features

    def forward(self, x):
        for i, n in enumerate(self.__model):
            x = n(x)
            if i == self._i:
                low = x
        return low, x

def mobilenetv3l(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="large")

def mobilenetv3s(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="small")


if __name__=="__main__":
    m=MNv3_backbone(pretrained=False,arch="large")
    x=torch.rand(1,3,512,512)
    y1,y2=m(x)
    print(y1.shape)
    print(y2.shape)

    m = MNv3_backbone(pretrained=False, arch="small")
    x = torch.rand(1, 3, 512, 512)
    y1, y2 = m(x)
    print(y1.shape)
    print(y2.shape)