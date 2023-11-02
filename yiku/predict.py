# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from AllLabs import DeeplabV3
from utils.download import download_from_url
from tempfile import TemporaryDirectory
import argparse,configparser
from utils import ASSETS
import torch
import os


def dir_predict(m,i,o,**kwargs):
    import os
    from tqdm import tqdm

    img_names = os.listdir(i)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(i, img_name)
            image = Image.open(image_path)
            r_image = m.detect_image(image)
            if not os.path.exists(o):
                os.makedirs(o)
            r_image.save(os.path.join(o, img_name))

def pic_predict(m,i,o,**kwargs):
    image = Image.open(i)
    r_image = m.detect_image(image)
    if not os.path.exists(o):
        os.makedirs(o)
    if bool(kwargs["show"]):
        r_image.show()
    else:
        r_image.save(os.path.join(o, str(Path(i).name)))

def http_predict(m,i,o,**kwargs):
    with TemporaryDirectory() as dirname:
        i=download_from_url(i,dirname)
        if Path(i).suffix.lower() in (".mp4",".mkv",".mov",".hevc"):
            video_predict(m,i,o,**kwargs)
        elif Path(i).suffix.lower() in (".png",".jpg",".jpeg","bmp", '.pbm', '.pgm', '.ppm', '.tif', '.tiff'):
            pic_predict(m,i,o,**kwargs)


def video_predict(m,i,o,**kwargs):
    show=kwargs["show"]
    video_fps = 25.0
    capture = cv2.VideoCapture(i)
    if o != "":
        if str(i).isdigit():
            video_save_path=os.path.join(o,"camera_%i.mp4"%i)
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


def auto_input_type(path:str):
    mode=None
    if path.isdigit():
        return "video_predict"
    if not path.startswith("http"):
        path=Path(path)
        if path.exists() and (not path.is_socket()):
            if path.is_dir():
                mode="dir_predict"
            elif path.is_file() :
                if str(path.suffix).lower() in (".png",".jpg",".jpeg","bmp", '.pbm', '.pgm', '.ppm', '.tif', '.tiff'):
                    mode = "pic_predict"
                elif str(path.suffix).lower() in (".mp4",".mkv",".mov",".hevc"):
                    mode = "video_predict"
                else:
                    raise NameError("Not a media file %s"%str(path))
        else:
            raise FileNotFoundError("%s" % str(path))
    else:
        mode="http_predict"

    return mode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.ini")
    parser.add_argument('-i', '--input', default=ASSETS/"amp-test.jpg",help="input to be inferenced ,Accept file dir cameraindex,and uri")
    parser.add_argument( '--show',action="store_true",help="flag to enable playback")
    parser.add_argument("-m", '--model', default=None, help=".pth model path to override config file")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    if os.path.exists(args.config):
        config.read(args.config, encoding="utf-8")
    else:
        config["base"] = {}
        config["advance"] = {}
    CONFIG_DIR = os.path.dirname(os.path.abspath(args.config))
    IMGSZ = config["base"].getint("image_size", 512)
    SAVE_PATH = os.path.join(CONFIG_DIR, config["base"].get("save_path"))
    if not os.path.isabs(SAVE_PATH):
        SAVE_PATH = os.path.join(CONFIG_DIR, SAVE_PATH)

    BACKBONE = config["base"].get("backbone", "hgnetv2l")
    PP = config["base"].get("header", "transformer")
    NUM_CLASSES = config["base"].getint("num_classes", 21)
    model_path = os.path.join(SAVE_PATH, "best_epoch_weights.pth")
    if args.model and os.path.isfile(str(args.model)):
        model_path=str(args.model)
    mode=auto_input_type(str(args.input))
    if mode:
        m = DeeplabV3(num_classes=NUM_CLASSES, backbone=BACKBONE, model_path=model_path
                            , pp=PP, cuda=torch.cuda.is_available(),input_shape=[IMGSZ,IMGSZ])
        func_se={"dir_predict":dir_predict,"pic_predict":pic_predict,"video_predict":video_predict,"http_predict":http_predict}
        o=Path(SAVE_PATH)/("result_%s"%mode)
        os.makedirs(o,exist_ok=True)
        i=str(args.input)
        o=str(o)
        func_se[mode](m,i,o,show=args.show)

if __name__ == "__main__":
    main()