from .VOCdataloader import VOCDataset
from .SimpleDataloader import SimpleDataset
from pathlib import Path
import os,glob
import torch
import numpy as np

def auto_ds_sel(input_shape, num_classes, train, dataset_path,**kwargs):
    voc=False
    for s in ["voc*","VOC*"]:
        for d in glob.glob(str(Path(dataset_path)/s)):
            if os.path.isdir(d):
                voc=True
                break
    if voc:
        return VOCDataset(input_shape, num_classes, train, dataset_path,**kwargs)
    if os.path.isdir(Path(dataset_path)/"images") and os.path.isdir(Path(dataset_path/"mask")):
        return SimpleDataset(input_shape, num_classes, train, dataset_path, **kwargs)


def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels