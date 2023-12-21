from .HarDN import HarDNet
def hardnet(**kwargs):
    m= HarDNet(num_classes=kwargs.get("num_classes"),pretrained=kwargs.get("pretrained",False))
    return m
    # if kwargs.get("pretrained",False):
    #     from yiku.PATH import WTS_STORAGE_DIR
    #     import torch
    #     with open(WTS_STORAGE_DIR/"hardnet.pth",mode="rb")as f:
    #         wts=torch.load(f,map_location=torch.device("cpu"))
    #         m.load_state_dict(wts,strict=False)
    #         return m
    # else:
    #     return m