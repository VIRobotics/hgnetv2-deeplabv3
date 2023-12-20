import requests
import re,os
from urllib.parse import urlparse
import pathlib
from yiku.PATH import WTS_STORAGE_DIR
from pip._vendor.rich.progress import (
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    Progress
)
class  IntegrityError(Exception):
    pass
def download_from_url(url,dir_path):
    a = urlparse(url)
    fname = os.path.basename(a.path)
    if os.path.isfile(os.path.join(dir_path,fname)):
        return os.path.join(dir_path,fname)
    response=requests.get(url,stream=True,allow_redirects=True)
    if "Content-Disposition" in response.headers.keys():
        d=response.headers["Content-Disposition"]
        pathlib.Path(os.path.join(dir_path,fname)).touch()
        fname = re.findall("filename=(.+)", d)[0]
    print("下载文件:%s"%fname)
    path=os.path.join(dir_path,fname)
    block_size = 1024  # 1 Kibibyte
    if 'Content-Encoding' in response.headers.keys():
        unlimited=True
    else:
        unlimited = False


    if unlimited:
        progress = Progress(
            SpinnerColumn(),
            "{task.description}",
            DownloadColumn(binary_units=True),
            TransferSpeedColumn(),
            TimeElapsedColumn()
        )
        task1 = progress.add_task("[red]Downloading %s" % fname,total=None)
        progress.start()
        with open(path, mode="wb") as f:
            for data in response.iter_content(block_size):
                progress.update(task1,advance=len(data))
                f.write(data)

        progress.stop()
        return

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    # progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    progress = Progress(
        SpinnerColumn(),
        "{task.description}",
        BarColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    task1 = progress.add_task("[red]Downloading %s"%fname, total=total_size_in_bytes)
    l=0
    progress.start()
    with open(path,mode="wb")as f:
        for data in response.iter_content(block_size):
            progress.update(task1,advance=len(data))
            f.write(data)
            l=l+len(data)
    progress.stop()
    if total_size_in_bytes != 0 and l != total_size_in_bytes:
        os.remove(path)
        raise IntegrityError

    return path


def download_weights(backbone, model_dir=WTS_STORAGE_DIR):
    import os
    from yiku.utils.download import download_from_url,IntegrityError

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
            'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'],
        "hardnet":["http://dl.aiblockly.com:8145/pretrained-model/seg/hardnet.pth"]
    }
    for i in range(6):
        download_urls[f"b{i}"]=[f"http://dl.aiblockly.com:8145/pretrained-model/seg/segformer_b{i}_backbone_weights.pth"]
    if backbone not in download_urls.keys():
        UserWarning("目前无法下载")
        return
    urls = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for url in urls:
        try:
            download_from_url(url, model_dir)
            break
        except (IntegrityError,ConnectionError):
            UserWarning("下载失败，重试")


if __name__=="__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        download_from_url("https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",d)