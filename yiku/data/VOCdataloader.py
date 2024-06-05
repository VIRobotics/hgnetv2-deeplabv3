import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from yiku.data.process import cvtColor, preprocess_input,resize_data,bias_rand,flip_data,blur_data,hsv_jiiter





class VOCDataset(Dataset):
    def __init__(self,  input_shape, num_classes, train, dataset_path,**kwargs):
        super(VOCDataset, self).__init__()
        self.dataset_path = dataset_path
        if train:
            with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
                annotation_lines = f.readlines()
        else:
            with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
                annotation_lines = f.readlines()
        self.__img=[]
        self.__label=[]
        for name in annotation_lines:
            name = name.split()[0]
            for sfx in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                if (os.path.isfile(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + sfx))
                        and os.path.isfile(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))):
                    self.__img.append(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + sfx))
                    self.__label.append(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
                    break
        self.annotation_lines = annotation_lines
        self.length = len(self.__img)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train

        self.blur = blur_data
        self.hsv_jitter = hsv_jiiter
        self.__dict__.update(kwargs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

            # -------------------------------#
            #   从文件中读取图像
            # -------------------------------#
        jpg = Image.open(self.__img[index])
        jpg.load()
        png = Image.open(self.__label[index])
        png.load()
            # -------------------------------#
            #   数据增强
            # -------------------------------#
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
            # -------------------------------------------------------#
            #   转化成one_hot的形式
            #   在这里需要+1是因为voc数据集有些标签具有白边部分
            #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
            # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        image, label, nw, nh = resize_data(label, image, iw, ih, w, h, getattr(self, "jitter_prop", 0.3))

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        image, label = flip_data(label, image, prop=getattr(self, "flip_prop", 0.5))

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(bias_rand(0, w - nw))
        dy = int(bias_rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # ------------------------------------------#
        #   高斯模糊
        # ------------------------------------------#
        if callable(self.blur):
            label, image_data = self.blur(label, image_data, prop=getattr(self, "blur_prop", 0.25))

        # ------------------------------------------#
        #   旋转
        # ------------------------------------------#
        rotate = bias_rand() < getattr(self, "rotation_prop", 0.25)
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
        if callable(self.hsv_jitter) and getattr(self, "hsv_jitter_enable", True):
            image_data = self.hsv_jitter(image_data, hue, sat, val)
        return image_data, label


# DataLoader中collate_fn使用

