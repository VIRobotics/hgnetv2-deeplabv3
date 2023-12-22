from torch import nn
import torch
from .PP_block import ConvBNReLU,ConvBNAct,ConvBN,SeparableConvBNReLU
import torch.nn.functional as F
from torch.autograd import Variable
class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias=False,
                act_type="leakyrelu"),
            ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_ChAtten_S(UAFM):
    """
    The UAFM with channel attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(
                2 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias=False,
                act_type="leakyrelu"),
            ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))
        self._scale = Variable(
            torch.ones([1],dtype=torch.float32)
            )
        self._scale.stop_gradient = True

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten_S(UAFM):
    """
    The UAFM with spatial attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                2, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFMMobile(UAFM):
    """
    Unified Attention Fusion Module for mobile.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_x = SeparableConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = SeparableConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)


class UAFMMobile_SpAtten(UAFM):
    """
    Unified Attention Fusion Module with spatial attention for mobile.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_x = SeparableConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = SeparableConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


def avg_reduce_hw(x):
    # Reduce hw by avg
    # Return cat([avg_pool_0, avg_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return torch.concat(res, dim=1)


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when dim=[2, 3], the torch.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = torch.max(x, dim=[2, 3], keepdim=True)

    if use_concat:
        res = torch.concat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.concat(res, dim=1)


def avg_reduce_channel(x):
    # Reduce channel by avg
    # Return cat([avg_ch_0, avg_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.mean(x, dim=1, keepdim=True)
    elif len(x) == 1:
        return torch.mean(x[0], dim=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.mean(xi, dim=1, keepdim=True))
        return torch.concat(res, dim=1)


def max_reduce_channel(x):
    # Reduce channel by max
    # Return cat([max_ch_0, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.max(x, dim=1, keepdim=True)
    elif len(x) == 1:
        return torch.max(x[0], dim=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.max(xi, dim=1, keepdim=True))
        return torch.concat(res, dim=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.amax(x, dim=1, keepdim=True)
    if use_concat:
        res = torch.concat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.concat(res, dim=1)


def cat_avg_max_reduce_channel(x):
    # Reduce hw by cat+avg+max
    assert isinstance(x, (list, tuple)) and len(x) > 1

    x = torch.concat(x, dim=1)

    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)
    res = torch.concat([mean_value, max_value], dim=1)

    return res