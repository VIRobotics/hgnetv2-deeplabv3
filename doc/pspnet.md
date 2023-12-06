
### 配置文件

```ini
[base]
arch = pspnet
```
记得清除掉配置文件里 lr 和优化器相关的配置选项 以使用默认的超参数

### 性能情况


| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [pspnet_mobilenetv2.pth](https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/pspnet_mobilenetv2.pth) | VOC-Val12 | 473x473| 68.59 | 
| VOC12+SBD | [pspnet_resnet50.pth](https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/pspnet_resnet50.pth) | VOC-Val12 | 473x473| 81.44 | 

#### 目前该Arch支持的主干网络有

ResNet50 Mobilenetv2



```ini
[base]
backbone = mobilenetv2
```

|   backbone    |
|:-------------:|
|   resnet50    |
|  mobilenetv2  |
| hgnetv2l(开发中) |


#### 目前该Arch支持的分割头有

配置文件中 此字段无效


### 文件下载



VOC拓展数据集的百度网盘如下：   
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng    
提取码: 44mk   
