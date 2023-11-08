
### 配置文件

```ini
[base]
arch = lab 
```
或者啥都不加

### 性能情况

|   训练数据集   |                                                             权值文件名称                                                              |   测试数据集   | 输入图片大小  | mIOU  | 
|:---------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------:|:-------:|:-----:| 
| VOC12+SBD | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val12 | 512x512 | 72.59 | 
| VOC12+SBD |    [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth)    | VOC-Val12 | 512x512 | 76.95 | 
| VOC12+SBD |                  [deeplab_hgnetv2.pth](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)                   | VOC-Val12 | 512x512 | 78.83 |
| VOC12+SBD |                  [translab_hgnetv2.pth](https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/tag/v0.0.2-beta)                 | VOC-Val12 | 512x512 | 80.23 |

#### 目前该Arch支持的主干网络有

MobileNetv2 MobileNetv3 XCeption HGNetv2(HGNet由百度开发，仪酷智能接入deeplab)，

YOLOv8(S和M尺寸，目前存在低mIOU的问题，不推荐)

```ini
[base]
backbone = hgnetv2l
```

|         backbone          |
|:-------------------------:|
|        mobilenetv2        |
| mobilenetv3s,mobilenetv3l |
|     hgnetv2x,hgnetv2l     |
|      yolov8s,yolov8m      |
|         xception          |

#### 目前该Arch支持的分割头有

官方Deeplabv3+的头（采用ASPP)

仪酷智能科技的TransLab头(采用AIFI Transformer) 

您可以自由的组合主干和分割头

```ini
[base]
header = transformer
```

|    header    |
|:------------:|
|     ASPP     |
| transformer  |


### 文件下载

比较新的deeplab_HGNetv2由仪酷智能科技提供 [链接](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)

```SHA256: D5DD6AB2556F87B8F03F12CCC14DCBEBADF01123003E1FBF3DB749D6477DBF8F```

训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 提取码: qqq4

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk