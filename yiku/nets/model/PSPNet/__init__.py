
def pspnet(**kwargs):
    if "backbone" in kwargs.keys():
        if kwargs["backbone"]=="resnet50":
            from .bubbliiiing_pspnet import pspNet
            return pspNet(kwargs["num_classes"],kwargs['downsample_factor'],pretrained=kwargs["pretrained"],backbone="resnet50")
        elif kwargs["backbone"]=="mobilenetv2":
            from .bubbliiiing_pspnet import pspNet
            return pspNet(kwargs["num_classes"],kwargs['downsample_factor'], pretrained=kwargs["pretrained"], backbone="mobilenet")

