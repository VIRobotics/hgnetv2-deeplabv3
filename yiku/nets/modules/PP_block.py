from torch import nn
import torch
import torch.nn.functional as F
from yiku.PATH import WTS_STORAGE_DIR
import numpy as np
try:
    from rich import print
except ImportError:
    import warnings

    warnings.filterwarnings('ignore', message="Setuptools is replacing distutils.", category=UserWarning)
    from pip._vendor.rich import print
def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    # if torch.get_device() == 'cpu' or os.environ.get(
    #         'PADDLESEG_EXPORT_STAGE') or 'xpu' in torch.get_device(
    #         ) or 'npu' in torch.get_device():
    #     return nn.BatchNorm2D(*args, **kwargs)
    # elif torch.distributed.ParallelEnv().nranks == 1:
    #     return nn.BatchNorm2D(*args, **kwargs)
    # else:
    #     return nn.SyncBatchNorm(*args, **kwargs)
    return nn.BatchNorm2d(*args,**kwargs)

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 **kwargs):
        super().__init__()


        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)
        self._relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

class ConvGNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding="same",
                 num_groups=32,
                 act_type=None,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if "data_format" in kwargs:
            data_format = kwargs["data_format"]
        else:
            data_format = "NCHW"
        self._group_norm = nn.GroupNorm(
            num_groups, out_channels, data_format=data_format)
        self._act_type = act_type
        if act_type is not None:
            self._act = Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        x = self._group_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 norm=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'

        self._norm = norm if norm is not None else None

        self._act_type = act_type
        if act_type is not None:
            self._act = Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        if self._norm is not None:
            x = self._norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

        self._act_type = act_type
        if act_type is not None:
            self._act = Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self._relu = Activation("ReLU")
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format,
            bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Module):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

from torch.nn import (ReLU6,ReLU,SiLU,SELU,LeakyReLU,Tanh,GELU,Hardswish,Hardshrink,PReLU,Hardtanh,Hardsigmoid,
    Sigmoid,Softmax)

class Activation(nn.Module):
    def __init__(self,act=None):
        super().__init__()
        if act is not None:
            d = {
                "relu": ReLU,
                "relu6": ReLU6,
                "silu": SiLU, "selu": SELU, "leakrelu": LeakyReLU,
                "tanh": Tanh
            }
            self.m=d[act.lower()]()
        else:
            self.m=None
    def forward(self,x):
        if self.m is not None:
            return self.m(x)
        else:
            return x
