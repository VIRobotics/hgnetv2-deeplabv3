from torch import nn
import torch
import numpy as np


def initBN_const(sublayer, w=1.0, b=0.0):
    if len(sublayer.weight.data.shape) < 2:
        nn.init.constant_(sublayer.weight.data.unsqueeze(0), val=w)
    else:
        nn.init.constant_(sublayer.weight.data, val=w)

    if len(sublayer.bias.data.shape) < 2:
        nn.init.constant_(sublayer.bias.data.unsqueeze(0), val=b)
    else:
        nn.init.constant_(sublayer.bias.data, val=b)


def kaiming_normal_(x):
    if len(x.data.shape) < 2:
        nn.init.kaiming_normal_(x.data.unsqueeze(0))
    else:
        nn.init.kaiming_normal_(x.data)


def trunc_normal_(x, std=.02):
    if len(x.data.shape) < 2:
        nn.init.trunc_normal_(x.data.unsqueeze(0), std=std)
    else:
        nn.init.trunc_normal_(x.data, std=std)


def zeros_(x):
    if len(x.data.shape) < 2:
        nn.init.zeros_(x.data.unsqueeze(0))
    else:
        nn.init.zeros_(x.data)


def ones_(x):
    if len(x.data.shape) < 2:
        nn.init.ones_(x.data.unsqueeze(0))
    else:
        nn.init.ones_(x.data)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    drop_prob = np.array(drop_prob)
    keep_prob = torch.from_numpy(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
