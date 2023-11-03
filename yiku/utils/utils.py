import numpy as np
from PIL import Image
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
        'mobilenet': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'],
        'xception': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'],
        'hgnetv2l': ['https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2l.pt',
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2l.pt"],
        "hgnetv2x": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2x.pt"],
        "yolov8s": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"],
        "yolov8m": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"],

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
