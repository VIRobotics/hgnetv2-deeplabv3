from yiku.utils.download import download_from_url,IntegrityError,DownloadError
import tempfile
from hashlib import sha256
download_urls = {
        'mobilenetv2': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
                        "http://dl.aiblockly.com:8145/pretrained-model/seg/mobilenet_v2.pth.tar"],
        'xception': ['https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/xception_pytorch_imagenet.pth"],
        'hgnetv2l': ['https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2l.pt',
                     "http://download.aiblockly.com/pretrained_wts/seg/hgnetv2l.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2l.pt"],
        "hgnetv2x": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",
                     "http://download.aiblockly.com/pretrained_wts/seg/hgnetv2x.pt",
                     "http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2x.pt"],
        "yolov8s": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"],
        "yolov8m": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"],
        "resnet50":["https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
                   "http://download.aiblockly.com/pretrained_wts/seg/resnet50-19c8e357.pth"],
        "vgg":["https://download.pytorch.org/models/vgg16-397923af.pth"],
        'mobilenetv3l': [
            'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'],
        'mobilenetv3s': [
            'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'],
        "hardnet":["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hardnet_backbone.pt",
                   "http://dl.aiblockly.com:8145/pretrained-model/seg/hardnet_backbone.pt"]
    }

def download_all_and_checksum():
    checksum = dict()
    for key in download_urls.keys():
        urls = download_urls[key]
        with tempfile.TemporaryDirectory() as tmpdirname:
            for url in urls:
                try:
                    path=download_from_url(url, tmpdirname)
                    with open(path,mode="rb")as f:
                        sum=sha256()
                        sum.update(f.read())
                        checksum[key]=sum.hexdigest()
                    break
                except (IntegrityError, ConnectionError,DownloadError):
                    UserWarning("下载失败，重试")
    return checksum
if __name__ == '__main__':
    import json
    with open("checksum.json",mode="w",encoding="utf8")as f:
        json.dump(download_all_and_checksum(),f,ensure_ascii=False,indent=4)