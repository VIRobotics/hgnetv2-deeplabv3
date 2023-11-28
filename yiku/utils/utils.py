import cv2
import numpy as np
from PIL import Image
from torch import nn
import torch
import os
from PATH import WTS_STORAGE_DIR
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    from pip._vendor.rich.table import Table
    from pip._vendor.rich.console import Console


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
        #return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    table = Table(title="ConfigurationTable参数详情")
    table.add_column("参数Arg", justify="right", style="cyan", no_wrap=True)
    table.add_column("值Value", style="magenta")
    for key, value in kwargs.items():
        table.add_row(key,str(value))
    console = Console()
    console.print(table)


def download_weights(backbone, model_dir=WTS_STORAGE_DIR):
    import os
    from utils.download import download_from_url,IntegrityError

    download_urls = {
        'mobilenetv2': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'],
        'xception': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'],
        'hgnetv2l': ['https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2l.pt',
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2l.pt"],
        "hgnetv2x": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2x.pt"],
        "yolov8s": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"],
        "yolov8m": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"],
        "resnet50":["https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth"],
        "vgg":["https://download.pytorch.org/models/vgg16-397923af.pth"],
        'mobilenetv3l': [
            'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'],
        'mobilenetv3s': [
            'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth']

    }
    urls = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for url in urls:
        try:
            download_from_url(url, model_dir)
            break
        except (IntegrityError,ConnectionError):
            UserWarning("下载失败，重试")

def fuse_conv_and_bn(module):
    module_output = module
    if isinstance(module, (nn.Sequential,)):
        print("[nn.Sequential]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(
                    module[idx + 1], nn.BatchNorm2d
            ):
                continue
            conv = module[idx]
            bn = module[idx + 1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                    conv.weight
                    * bn.weight[:, None, None, None]
                    * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
            conv.bias.data = (
                                     conv.bias - bn.running_mean
                             ) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(name, fuse_conv_and_bn(child))
    del module
    return module_output

import requests
def get_github_assets(repo='ultralytics/assets', version='latest', retry=False):
    """Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov8s.pt', ...])."""
    if version != 'latest':
        version = f'tags/{version}'  # i.e. tags/v6.2
    url = f'https://api.github.com/repos/{repo}/releases/{version}'
    r = requests.get(url)  # github api
    if r.status_code != 200 and r.reason != 'rate limit exceeded' and retry:  # failed and not 403 rate limit exceeded
        r = requests.get(url)  # try again
    if r.status_code != 200:
        print(f'⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}')
        return '', []
    data = r.json()
    return data['tag_name'], [x['name'] for x in data['assets']]