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