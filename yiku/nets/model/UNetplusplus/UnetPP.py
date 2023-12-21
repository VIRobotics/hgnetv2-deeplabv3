from torch import nn
import torch
class UNetPlusPlus(nn.Module):
    """

    The original article refers to
    Zongwei Zhou, et, al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    (https://arxiv.org/abs/1807.10165).

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): The channel number of input image. Default: 3.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
        is_ds (bool): use deep supervision or not. Default: True
        """

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 use_deconv=False,
                 align_corners=False,
                 pretrained=None,
                 is_ds=True):
        super(UNetPlusPlus, self).__init__()
        self.pretrained = pretrained
        self.is_ds = is_ds
        channels = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = DoubleConv(in_channels, channels[0])
        self.conv1_0 = DoubleConv(channels[0], channels[1])
        self.conv2_0 = DoubleConv(channels[1], channels[2])
        self.conv3_0 = DoubleConv(channels[2], channels[3])
        self.conv4_0 = DoubleConv(channels[3], channels[4])

        self.up_cat0_1 = UpSampling(
            channels[1],
            channels[0],
            n_cat=2,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat1_1 = UpSampling(
            channels[2],
            channels[1],
            n_cat=2,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat2_1 = UpSampling(
            channels[3],
            channels[2],
            n_cat=2,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat3_1 = UpSampling(
            channels[4],
            channels[3],
            n_cat=2,
            use_deconv=use_deconv,
            align_corners=align_corners)

        self.up_cat0_2 = UpSampling(
            channels[1],
            channels[0],
            n_cat=3,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat1_2 = UpSampling(
            channels[2],
            channels[1],
            n_cat=3,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat2_2 = UpSampling(
            channels[3],
            channels[2],
            n_cat=3,
            use_deconv=use_deconv,
            align_corners=align_corners)

        self.up_cat0_3 = UpSampling(
            channels[1],
            channels[0],
            n_cat=4,
            use_deconv=use_deconv,
            align_corners=align_corners)
        self.up_cat1_3 = UpSampling(
            channels[2],
            channels[1],
            n_cat=4,
            use_deconv=use_deconv,
            align_corners=align_corners)

        self.up_cat0_4 = UpSampling(
            channels[1],
            channels[0],
            n_cat=5,
            use_deconv=use_deconv,
            align_corners=align_corners)

        self.out_1 = nn.Conv2d(channels[0], num_classes, 1, 1, 0)
        self.out_2 = nn.Conv2d(channels[0], num_classes, 1, 1, 0)
        self.out_3 = nn.Conv2d(channels[0], num_classes, 1, 1, 0)
        self.out_4 = nn.Conv2d(channels[0], num_classes, 1, 1, 0)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            pass
        else:
            for sublayer in self.modules():
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight.data)
                elif isinstance(sublayer, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if len(sublayer.weight.data.shape)<2:
                        nn.init.kaiming_normal_(sublayer.weight.data.unsqueeze(0))
                    else:
                        nn.init.kaiming_normal_(sublayer.weight.data)

    def forward(self, inputs):
        # 0 down
        X0_0 = self.conv0_0(inputs)  # n,32,h,w
        pool_0 = self.pool(X0_0)  # n,32,h/2,w/2
        X1_0 = self.conv1_0(pool_0)  # n,64,h/2,w/2
        pool_1 = self.pool(X1_0)  # n,64,h/4,w/4
        X2_0 = self.conv2_0(pool_1)  # n,128,h/4,w/4
        pool_2 = self.pool(X2_0)  # n,128,h/8,n/8
        X3_0 = self.conv3_0(pool_2)  # n,256,h/8,w/8
        pool_3 = self.pool(X3_0)  # n,256,h/16,w/16
        X4_0 = self.conv4_0(pool_3)  # n,512,h/16,w/16

        # 1 up+concat
        X0_1 = self.up_cat0_1(X1_0, X0_0)  # n,32,h,w
        X1_1 = self.up_cat1_1(X2_0, X1_0)  # n,64,h/2,w/2
        X2_1 = self.up_cat2_1(X3_0, X2_0)  # n,128,h/4,w/4
        X3_1 = self.up_cat3_1(X4_0, X3_0)  # n,256,h/8,w/8

        # 2 up+concat
        X0_2 = self.up_cat0_2(X1_1, X0_0, X0_1)  # n,32,h,w
        X1_2 = self.up_cat1_2(X2_1, X1_0, X1_1)  # n,64,h/2,w/2
        X2_2 = self.up_cat2_2(X3_1, X2_0, X2_1)  # n,128,h/4,w/4

        # 3 up+concat
        X0_3 = self.up_cat0_3(X1_2, X0_0, X0_1, X0_2)  # n,32,h,w
        X1_3 = self.up_cat1_3(X2_2, X1_0, X1_1, X1_2)  # n,64,h/2,w/2

        # 4 up+concat
        X0_4 = self.up_cat0_4(X1_3, X0_0, X0_1, X0_2, X0_3)  # n,32,h,w

        # out conv1*1
        out_1 = self.out_1(X0_1)  # n,num_classes,h,w
        out_2 = self.out_2(X0_2)  # n,num_classes,h,w
        out_3 = self.out_3(X0_3)  # n,num_classes,h,w
        out_4 = self.out_4(X0_4)  # n,num_classes,h,w

        output = (out_1 + out_2 + out_3 + out_4) / 4

        if self.is_ds:
            return [output]
        else:
            return [out_4]


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size=3,
                 stride=1,
                 padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, filter_size, stride, padding),
            nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, inputs):
        conv = self.conv(inputs)

        return conv


class UpSampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_cat,
                 use_deconv=False,
                 align_corners=False):
        super(UpSampling, self).__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=align_corners),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))

        self.conv = DoubleConv(n_cat * out_channels, out_channels)

    def forward(self, high_feature, *low_features):
        features = [self.up(high_feature)]
        for feature in low_features:
            features.append(feature)
        cat_features = torch.concat(features, dim=1)
        out = self.conv(cat_features)

        return out
