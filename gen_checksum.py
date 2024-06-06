from yiku.utils.download import download_by_aria2,IntegrityError,DownloadError
from yiku.PATH import WTS_STORAGE_DIR
from hashlib import sha256
download_urls = {
        'mobilenetv2': [
            'https://gitee.com/yiku-ai/hgnetv2-deeplabv3/releases/download/asset/mobilenet_v2.pth.tar',
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
            "http://yiku.hopto.org:8145/pretrained-model/seg/mobilenet_v2.pth.tar",
            "http://assets.virobotics.net/pretrained_model/seg/mobilenet_v2.pth.tar"],
        'xception': [
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
            "http://yiku.hopto.org:8145/pretrained-model/seg/xception_pytorch_imagenet.pth"],
        'hgnetv2l': ["http://assets.virobotics.net/pretrained_model/seg/hgnetv2l.pt",
                     'https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2l.pt',
                     "http://download.aiblockly.com/pretrained_wts/seg/hgnetv2l.pt",
                     "http://yiku.hopto.org:8145/pretrained-model/seg/hgnetv2l.pt"],
        "hgnetv2x": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hgnetv2x.pt",
                     "http://assets.virobotics.net/pretrained_model/seg/hgnetv2x.pt",
                     "http://yiku.hopto.org:8145/pretrained-model/seg/hgnetv2x.pt"],
        "resnet50": ["https://gitee.com/yiku-ai/hgnetv2-deeplabv3/releases/download/asset/resnet50-19c8e357.pth",
                     "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
                     "http://assets.virobotics.net/pretrained_model/seg/resnet50-19c8e357.pth"],
        "vgg": ["https://download.pytorch.org/models/vgg16-397923af.pth"],
        'mobilenetv3l': [
            'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'],
        'mobilenetv3s': [
            'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'],
        "hardnet": ["https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/hardnet_backbone.pt",
                    "http://yiku.hopto.org:8145/pretrained-model/seg/hardnet_backbone.pt"],
        "yolov8s": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"],
        "yolov8m": ["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt"]
    }
for i in range(6):
    download_urls[f"b{i}"] = [
        f"https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/segformer_b{i}_backbone_weights.pth"
        , f"http://yiku.hopto.org:8145/pretrained-model/seg/segformer_b{i}_backbone_weights.pth",
        f"http://assets.virobotics.net/pretrained_model/seg/segformer_b{i}_backbone_weights.pth"]

def download_all_and_checksum():
    checksum = dict()
    for key in download_urls.keys():
        urls = download_urls[key]
        path,checksum[key]=download_by_aria2(urls,WTS_STORAGE_DIR)
    return checksum
if __name__ == '__main__':
    import json
    with open("checksum.json",mode="w",encoding="utf8")as f:
        json.dump(download_all_and_checksum(),f,ensure_ascii=False,indent=4)