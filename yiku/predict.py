# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from yiku.model_warp import Wrapper
from utils.download import download_from_url
from tempfile import TemporaryDirectory
import argparse, configparser
from utils import ASSETS
import torch
import os

try:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn, track,
        TimeElapsedColumn,
        TimeRemainingColumn)
    from rich import print
except ImportError:
    import warnings

    warnings.filterwarnings('ignore', message="Setuptools is replacing distutils.", category=UserWarning)
    from pip._vendor.rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn, track,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from pip._vendor.rich import print


def dir_predict(m, i, o, **kwargs):
    import os

    img_names = os.listdir(i)
    for img_name in track(img_names, description="Predicting"):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(i, img_name)
            image = Image.open(image_path)
            r_image = m.detect_image(image)
            if not os.path.exists(o):
                os.makedirs(o)
            r_image.save(os.path.join(o, img_name))


def pic_predict(m, i, o, **kwargs):
    image = Image.open(i)
    r_image = m.detect_image(image)
    if not os.path.exists(o):
        os.makedirs(o)
    if bool(kwargs["show"]):
        r_image.show()
    else:
        r_image.save(os.path.join(o, str(Path(i).name)))


def http_predict(m, i, o, **kwargs):
    with TemporaryDirectory() as dirname:
        i = download_from_url(i, dirname)
        if Path(i).suffix.lower() in (".mp4", ".mkv", ".mov", ".hevc"):
            video_predict(m, i, o, **kwargs)
        elif Path(i).suffix.lower() in (".png", ".jpg", ".jpeg", "bmp", '.pbm', '.pgm', '.ppm', '.tif', '.tiff'):
            pic_predict(m, i, o, **kwargs)


def video_predict(m, i, o, **kwargs):
    show = kwargs["show"]
    video_fps = 25.0
    capture = cv2.VideoCapture(i)
    if o != "":
        if str(i).isdigit():
            video_save_path = os.path.join(o, "camera_%i.mp4" % i)
        else:
            video_save_path = os.path.join(o, str(Path(i).name))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(m.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if show:
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


def auto_input_type(path: str):
    mode = None
    if path.isdigit():
        return "video_predict"
    if not path.startswith("http"):
        path = Path(path)
        if path.exists() and (not path.is_socket()):
            if path.is_dir():
                mode = "dir_predict"
            elif path.is_file():
                if str(path.suffix).lower() in (
                ".png", ".jpg", ".jpeg", "bmp", '.pbm', '.pgm', '.ppm', '.tif', '.tiff'):
                    mode = "pic_predict"
                elif str(path.suffix).lower() in (".mp4", ".mkv", ".mov", ".hevc"):
                    mode = "video_predict"
                else:
                    raise NameError("Not a media file %s" % str(path))
        else:
            raise FileNotFoundError("%s" % str(path))
    else:
        mode = "http_predict"

    return mode


def get_miou(miou_mode, pred_dir, m: Wrapper, ds_dir, num_classes, name_classes):
    image_ids = open(os.path.join(ds_dir, "VOC2007/ImageSets/Segmentation/val.txt"),
                     'r').read().splitlines()
    gt_dir = os.path.join(ds_dir, "VOC2007/SegmentationClass/")
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = m
        print("Load model done.")

        print("Get predict result.")
        for image_id in track(image_ids):
            image_path = os.path.join(ds_dir, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        from utils.utils_metrics import compute_mIoU
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.ini")
    parser.add_argument('-i', '--input', default=ASSETS / "amp-test.jpg",
                        help="input to be inferenced ,Accept file dir cameraindex,and uri")
    parser.add_argument('--show', action="store_true", help="flag to enable playback")
    parser.add_argument("-m", '--model', default=None, help=".pth model path to override config file")
    parser.add_argument('--get-miou', type=int, default=-1, help="-1,0,1,2 -1:disable ")
    parser.add_argument('--mix_type', type=int, default=0, help="0,1,2")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    if os.path.exists(args.config):
        config.read(args.config, encoding="utf-8")
    else:
        config["base"] = {}
        config["advance"] = {}
    CONFIG_DIR = os.path.dirname(os.path.abspath(args.config))
    IMGSZ = config["base"].getint("image_size", 512)
    SAVE_PATH = config["base"].get("save_path", "save")
    ARCH = config["base"].get("arch", "lab")
    if not os.path.isabs(SAVE_PATH):
        SAVE_PATH = os.path.join(CONFIG_DIR, SAVE_PATH)

    BACKBONE = config["base"].get("backbone", "hgnetv2l")
    PP = config["base"].get("header", "transformer")
    NUM_CLASSES = config["base"].getint("num_classes", 21)
    model_path = os.path.join(SAVE_PATH, "best.pth")
    mix_type = int(args.mix_type)
    DATASET_PATH = config["base"].get("dataset_path", 'VOCdevkit')
    if not os.path.isabs(DATASET_PATH):
        DATASET_PATH = os.path.join(CONFIG_DIR, DATASET_PATH)
    if args.model and os.path.isfile(str(args.model)):
        model_path = str(args.model)
    m = Wrapper(num_classes=NUM_CLASSES, backbone=BACKBONE, model_path=model_path
                , pp=PP, cuda=torch.cuda.is_available(), input_shape=[IMGSZ, IMGSZ], arch=ARCH, mix_type=mix_type)

    if args.get_miou in [0, 1, 2]:
        o = Path(SAVE_PATH) / ("result_%s" % "get_miou")
        os.makedirs(o, exist_ok=True)
        get_miou(args.get_miou, o, m, DATASET_PATH, num_classes=NUM_CLASSES, name_classes=["+"] * NUM_CLASSES)
        return
    mode = auto_input_type(str(args.input))
    if mode:
        func_se = {"dir_predict": dir_predict, "pic_predict": pic_predict, "video_predict": video_predict,
                   "http_predict": http_predict}
        o = Path(SAVE_PATH) / ("result_%s" % mode)
        os.makedirs(o, exist_ok=True)
        i = str(args.input)
        o = str(o)
        func_se[mode](m, i, o, show=args.show)


if __name__ == "__main__":
    main()
