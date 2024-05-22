import cv2
import numpy as np
from PIL import Image
from torch import nn
import torch
import os
from yiku.PATH import WTS_STORAGE_DIR

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    from pip._vendor.rich.table import Table
    from pip._vendor.rich.console import Console


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def show_config(**kwargs):
    table = Table(title="ConfigurationTable参数详情")
    table.add_column("参数Arg", justify="right", style="cyan", no_wrap=True)
    table.add_column("值Value", style="magenta")
    for key, value in kwargs.items():
        table.add_row(key, str(value))
    console = Console()
    console.print(table)


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
