from .HarDN import HarDNet
def hardnet(**kwargs):
    return HarDNet(num_classes=kwargs.get("num_classes"))