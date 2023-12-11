import time

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from yiku.nets.labs import Labs
from collections import OrderedDict
from tempfile import TemporaryDirectory
import argparse,configparser
from utils import ASSETS
import torch
import os
from yiku.utils.utils import resize_image,cvtColor

try:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,track,
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
        SpinnerColumn,track,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn
    )
    from pip._vendor.rich import print


def preprocess_input(image):
    image /= 255.0
    return image
def get_featureMap(m:Labs,ds_dir,mode="val",sz=512):
    image_ids = open(os.path.join(ds_dir, f"VOC2007/ImageSets/Segmentation/{mode}.txt"),
                     'r').read().splitlines()
    gt_dir = os.path.join(ds_dir, "VOC2007/SegmentationClass/")
    print("Get featuremap cache.")
    m.save_featuremap=True
    for image_id in track(image_ids):
        image_path = os.path.join(ds_dir, "VOC2007/JPEGImages/" + image_id + ".jpg")
        gt_img=os.path.join(gt_dir, image_id + ".png")
        image = Image.open(image_path)
        png=Image.open(gt_img)
        image=cvtColor(image)
        png=cvtColor(png)
        image, _, _ = resize_image(image, (sz,sz))
        png, _, _ = resize_image(png, (sz, sz),mode="L")
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        images = torch.from_numpy(image_data)
        images = images.cuda()
        _ = m(images)
        fm=m.featuremap_result
        with open(os.path.join(ds_dir, "VOC2007/JPEGImages/" + image_id + ".fm"),mode="wb")as f:
            torch.save(fm,f)

    print("Get predict result done.")
