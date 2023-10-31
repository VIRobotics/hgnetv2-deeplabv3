import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from nets.modules import SeparableConv2d,ASBlock
from PATH import WTS_STORAGE_DIR





class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, downsample_factor,bn_mom=0.0003):
        self.low_ch = 256
        self.feature_ch = 2048
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('Xception.py: output stride=%d is not supported.' % os)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        # do relu here

        self.block1 = ASBlock(64, 128, 2)
        self.block2 = ASBlock(128, 256, stride_list[0], inplace=False)
        self.block3 = ASBlock(256, 728, stride_list[1])

        rate = 16 // downsample_factor
        self.block4 = ASBlock(728, 728, 1, atrous=rate)
        self.block5 = ASBlock(728, 728, 1, atrous=rate)
        self.block6 = ASBlock(728, 728, 1, atrous=rate)
        self.block7 = ASBlock(728, 728, 1, atrous=rate)

        self.block8 = ASBlock(728, 728, 1, atrous=rate)
        self.block9 = ASBlock(728, 728, 1, atrous=rate)
        self.block10 = ASBlock(728, 728, 1, atrous=rate)
        self.block11 = ASBlock(728, 728, 1, atrous=rate)

        self.block12 = ASBlock(728, 728, 1, atrous=rate)
        self.block13 = ASBlock(728, 728, 1, atrous=rate)
        self.block14 = ASBlock(728, 728, 1, atrous=rate)
        self.block15 = ASBlock(728, 728, 1, atrous=rate)

        self.block16 = ASBlock(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block17 = ASBlock(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block18 = ASBlock(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block19 = ASBlock(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

        self.block20 = ASBlock(728, 1024, stride_list[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        return low_featrue_layer, x


def load_url(url, model_dir=WTS_STORAGE_DIR, map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def xception(pretrained=True, downsample_factor=16):
    model = Xception(downsample_factor=downsample_factor)
    if pretrained:
        model.load_state_dict(load_url(
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'),
                              strict=False)
    return model