import os.path

from torch import nn
from ultralytics.models import YOLO, RTDETR
import sys
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights,mobilenet_v3_large,MobileNet_V3_Large_Weights


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
class YOLOv8_backbone(nn.Module):   #### Not Recommand (Low mIOU)
    def __init__(self, *args, **kwargs):
        if "model" not in kwargs.keys():
            model = "yolov8s-cls.pt"
        else:
            model = kwargs["model"]
            kwargs.__delitem__("model")
        super().__init__(*args, **kwargs)
        m = YOLO(os.path.join("./model_data/",model))
        self.__model = m.model.model
        arch=model[6]
        self.__dict__.update(**{"s": {"feature_ch": 512, "low_ch": 128},
                                "m": {"feature_ch": 768, "low_ch": 192}
                                }[arch])

    def forward(self, x):
        for i, n in enumerate(self.__model):
            x = n(x)
            if i == len(self.__model) - 1 - 2:
                func = nn.Upsample(None, 2, 'nearest')
                y1 = func(x)
                break # Get the featuremap
            if i == 4:
                func = nn.Upsample(None, 2, 'nearest')
                low = func(x) # low feature
        return low, y1


class HG_backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        pretrained =bool(kwargs.get("pretrained",True))
        arch =str(kwargs.get("arch","s"))[0]
        for key in list(kwargs.keys()):
            kwargs.__delitem__(key)
        super().__init__(*args, **kwargs)
        m=RTDETR(f"rtdetr-{arch}.yaml")
        if pretrained:
            m.load(f"./model_data/rtdetr-{arch}.pt")
        self.__model = m.model.model
        self.__dict__.update(**{"l":{"feature_ch":2048,"low_ch":512},
                                "x":{"feature_ch":2048,"low_ch":512}
                                }[arch])

    def forward(self, x):
        for i, n in enumerate(self.__model):
            x = n(x)
            if x.shape[1] == 2048:# Get the featuremap
                break
            if i == 3:
                low = x # low feature

        return low, x

def mobilenetv3l(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="large")

def mobilenetv3s(pretrained=True, **kwargs):
    return MNv3_backbone(pretrained=pretrained,arch="small")

def hgnetv2l(pretrained=True, **kwargs):
    return HG_backbone(pretrained=pretrained,arch="l")

def hgnetv2x(pretrained=True, **kwargs):
    return HG_backbone(pretrained=pretrained,arch="x")

def yolov8s(pretrained=True, **kwargs):
    return YOLOv8_backbone(model="yolov8s-cls.pt")

def yolov8m(pretrained=True, **kwargs):
    return YOLOv8_backbone(model="yolov8m-cls.pt")