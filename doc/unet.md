
### 配置文件

```ini
[base]
arch = unet
```
记得清除掉配置文件里 lr 和优化器相关的配置选项 以使用默认的超参数

### 性能情况
**unet并不适合VOC此类数据集，其更适合特征少，需要浅层特征的医药数据集之类的。**

| 训练数据集 |                                                    权值文件名称                                                     | 测试数据集 | 输入图片大小 | mIOU  | 
| :-----: |:-------------------------------------------------------------------------------------------------------------:| :------: | :------: |:-----:| 
| VOC12+SBD |    [unet_vgg_voc.pth](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_vgg_voc.pth)      | VOC-Val12 | 512x512| 58.78 | 
| VOC12+SBD | [unet_resnet_voc.pth]([http://mirror.lan](http://dl.aiblockly.com:8145)/pretrained-model/seg/resnet50-unet.pth) | VOC-Val12 | 512x512| 72.09 |
| VOC12+SBD |  [unet_hgnetv2l_voc.pth](http://dl.aiblockly.com:8145/pretrained-model/seg/hgnetv2l-unet.pth)                   | VOC-Val12 | 512x512| 80.33 | 

#### 目前该Arch支持的主干网络有

ResNet50 VGG  HGNetv2，



```ini
[base]
backbone = hgnetv2l
```

| backbone |
|:--------:|
| resnet50 |
|   vgg    |
| hgnetv2l |


#### 目前该Arch支持的分割头有

配置文件中 此字段无效


### 文件下载

训练所需的权值可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A    
提取码: 6n2c   

VOC拓展数据集的百度网盘如下：   
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng    
提取码: 44mk   
