import cv2
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
        #return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


    # ---------------------------------------------------#

import numpy as np
from PIL import Image
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size,mode="RGB"):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    if mode=="L":
        color=(0)
    else:
        color=(128, 128, 128)
    new_image = Image.new(mode, size, color)
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh

def preprocess_input(image):
    image /= 255.0
    return image


def bias_rand(a=0.0, b=1.):
    return np.random.rand() * (b - a) + a

def flip_data(label,image,*args,prop=0.5):
    flip = bias_rand() < prop
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image,label



def blur_data(label,image,*args,prop=0.25):
    blur = bias_rand() < prop
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return label,image
def hsv_jiiter(image_data,h,s,v):
    # ---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    # ---------------------------------#
    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    # ---------------------------------#
    #   将图像转到HSV上
    # ---------------------------------#
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    # ---------------------------------#
    #   应用变换
    # ---------------------------------#
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

def resize_data(label,image,iw,ih,w,h,jitter):
    new_ar = iw / ih * bias_rand(1 - jitter, 1 + jitter) / bias_rand(1 - jitter, 1 + jitter)
    scale = bias_rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)
    label = label.resize((nw, nh), Image.NEAREST)
    return image,label,nw,nh