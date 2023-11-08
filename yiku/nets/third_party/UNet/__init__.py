def UNet(**kwargs):
    if "backbone" in kwargs.keys():
        if kwargs["backbone"]=="resnet50":
            from .bubbliiiing_unet import Unet
            return Unet(kwargs["num_classes"],pretrained=kwargs["pretrained"],backbone="resnet50")
        elif kwargs["backbone"]=="vgg":
            from .bubbliiiing_unet import Unet
            return Unet(kwargs["num_classes"], pretrained=kwargs["pretrained"], backbone="vgg")
    from .hgnetv2_unet import UNet as Unet
    return Unet(kwargs["num_classes"], pretrained=kwargs["pretrained"])


