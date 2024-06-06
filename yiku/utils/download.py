import json
import sys
import  subprocess
import requests
import re, os
from urllib.parse import urlparse
import pathlib
from yiku.PATH import WTS_STORAGE_DIR, ASSETS
from rich.progress import (
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    Progress
)
import hashlib

def get_contry():
    url="https://searchplugin.csdn.net/api/v1/ip/get"
    resp=requests.get(url)
    data=resp.json()
    addr=data["data"]["address"].split()[0]
    return addr

class IntegrityError(Exception):
    pass


class DownloadError(Exception):
    pass

def check_file_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, mode="rb") as f:
        while True:
            chunk = f.read(1024 * 16)
            if not chunk:
                break
            else:
                h.update(chunk)
        return h.hexdigest()
def download_by_aria2(urls:list, dir_path,sha256=None):
    a = urlparse(urls[0])
    fname = os.path.basename(a.path)
    if os.path.isfile(os.path.join(dir_path, fname)):
        return os.path.join(dir_path, fname), check_file_sha256(os.path.join(dir_path, fname))
    else:
        if "win" in sys.platform.lower():
            exec=ASSETS/"exec"/"aria2c.exe"
        else:
            exec ="aria2c"
        target_name=os.path.join(dir_path, fname)
        u=""
        for url in urls:
            u=u+'"'+url+'"  '
        if sha256:
            cmd=f"{exec}  -d {dir_path} -s 8 -x 8 --checksum=sha-256={sha256} {u}"
        else:
            cmd = f'"{exec}"  -d "{dir_path}" -s 8 -x 8 {u}'
        pipe = subprocess.Popen(cmd, shell=True, stdout=sys.stdout,stderr=sys.stdout)
        pipe.wait()
        return target_name,check_file_sha256(target_name)


def download_from_url(url, dir_path):
    a = urlparse(url)
    fname = os.path.basename(a.path)
    if os.path.isfile(os.path.join(dir_path, fname)):
        return os.path.join(dir_path, fname), check_file_sha256(os.path.join(dir_path, fname))
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
    except (requests.exceptions.SSLError) as e:
        raise DownloadError
    if "Content-Disposition" in response.headers.keys():
        d = response.headers["Content-Disposition"]
        pathlib.Path(os.path.join(dir_path, fname)).touch()
        fname = re.findall("filename=(.+)", d)[0]
    print("下载文件:%s" % fname)
    path = os.path.join(dir_path, fname)
    block_size = 1024  # 1 Kibibyte
    if 'Content-Encoding' in response.headers.keys():
        unlimited = True
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
        task1 = progress.add_task("[red]Downloading %s" % fname, total=None)
        progress.start()
        h=hashlib.sha256()
        with open(path, mode="wb") as f:
            for data in response.iter_content(block_size):
                progress.update(task1, advance=len(data))
                f.write(data)
                h.update(data)

        progress.stop()
        return None, h.hexdigest()

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
    task1 = progress.add_task("[red]Downloading %s" % fname, total=total_size_in_bytes)
    l = 0
    progress.start()
    h=hashlib.sha256()
    with open(path, mode="wb") as f:
        for data in response.iter_content(block_size):
            progress.update(task1, advance=len(data))
            f.write(data)
            h.update(data)
            l = l + len(data)
    progress.stop()
    if total_size_in_bytes != 0 and l != total_size_in_bytes:
        os.remove(path)
        raise IntegrityError

    return path, h.hexdigest()


def download_weights(backbone, model_dir=WTS_STORAGE_DIR):
    #is_China="中国" in get_contry()
    import os
    from yiku.utils.download import download_from_url, IntegrityError

    download_urls = {
        'mobilenetv2': [
            'https://gitee.com/yiku-ai/hgnetv2-deeplabv3/releases/download/asset/mobilenet_v2.pth.tar',
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
            "http://dl.aiblockly.com:8145/pretrained-model/seg/mobilenet_v2.pth.tar",
            "http://assets.virobotics.net/pretrained_model/seg/mobilenet_v2.pth.tar"],
        'xception': [
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
            "http://dl.aiblockly.com:8145/pretrained-model/seg/xception_pytorch_imagenet.pth"],
        'hgnetv2l': ["http://assets.virobotics.net/pretrained_model/seg/hgnetv2l.pt",
                     'https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2l.pt',
                     "http://download.aiblockly.com/pretrained_wts/seg/hgnetv2l.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2l.pt"],
        "hgnetv2x": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",
                     "http://assets.virobotics.net/pretrained_model/seg/hgnetv2x.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2x.pt"],
        "yolov8s": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"],
        "yolov8m": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"],
        "resnet50": ["https://gitee.com/yiku-ai/hgnetv2-deeplabv3/releases/download/asset/resnet50-19c8e357.pth",
                     "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
                     "http://assets.virobotics.net/pretrained_model/seg/resnet50-19c8e357.pth"],
        "vgg": ["https://download.pytorch.org/models/vgg16-397923af.pth"],
        'mobilenetv3l': [
            'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'],
        'mobilenetv3s': [
            'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'],
        "hardnet": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hardnet_backbone.pt",
                    "http://dl.aiblockly.com:8145/pretrained-model/seg/hardnet_backbone.pt"]
    }
    for i in range(6):
        download_urls[f"b{i}"] = [
            f"https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/segformer_b{i}_backbone_weights.pth"
            , f"http://dl.aiblockly.com:8145/pretrained-model/seg/segformer_b{i}_backbone_weights.pth",
            f"http://assets.virobotics.net/pretrained_model/seg/segformer_b{i}_backbone_weights.pth"]
    if backbone not in download_urls.keys():
        UserWarning("目前无法下载")
        return
    urls = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(ASSETS / "checksum.json", 'rb') as fj:
        sum_real = json.load(fj)[backbone]
        p,s=download_by_aria2(urls,model_dir,sha256=None)
        if sum_real != s:
            os.remove(p)
            raise IntegrityError(f"网络原因导致的文件{p}损坏\n预期的校验{sum_real}\n实际的校验{s}")
        return

    # for url in urls:
    #     try:
    #         path, hash = download_from_url(url, model_dir)
    #         with open(ASSETS / "checksum.json", 'rb') as fj:
    #             sum_real = json.load(fj)[backbone]
    #         if sum_real == hash:
    #             break
    #         else:
    #             UserWarning("数据校验失败，重试")
    #     except (IntegrityError, ConnectionError, DownloadError):
    #         UserWarning("下载失败，重试")


if __name__ == "__main__":
    print(download_weights("hgnetv2l"))
