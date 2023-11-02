## Torch版的TransLab和DeepLabv3+全家桶
---

### 目录

1. [仓库更新 Top News](#仓库更新)
2. [相关仓库 Related code](#相关仓库)
3. [性能情况 Performance](#性能情况)
4. [所需环境 Environment](#所需环境)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 miou](#评估步骤)
9. [参考资料 Reference](#Reference)

## Top News

**`2023-11`** : **使用命令行工具训练**

~~”不行啊，每次都要改代码，感觉不如OO啊“~~

您还在为改python文件烦恼吗，您还在为乱哄哄的文件夹烦恼吗，赶快使用
`pip install git+https://gitee.com/yiku-ai/hgnetv2-deeplabv3` 安装吧，装了你不吃亏，装了你不上当。


**`2023-09`** : **新增TransLab分割头，可以通过设置pp参数切换**

TransLab是一款由仪酷智能科技有限公司开发的分割头，在这款分割头里面，我们将DeepLabv3基于传统卷积的空洞卷积 换成了基于Transformer的AIFI模块

~~玩Transformer玩的~~

**`2023-08`**:**在原作者基础上添加多个新款Backbone（HGNetv2,yolov8系列）**

如果在 新模型（HGNetv2 YOLOv8 MobileNetv3)有疑问或者建议 欢迎issue和PR

仪酷LabView工业AI推理插件工具包已经支持此项目包括最新主干在内的模型

如果需要原版代码 请访问https://github.com/bubbliiiing/deeplabv3-plus-pytorch

**`2022-04`**:**支持多GPU训练。**

**`2022-03`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/deeplabv3-plus-pytorch/tree/bilibili

**`2020-08`**:**创建仓库、支持多backbone、支持数据miou评估、标注数据处理、大量注释等。**

## 相关仓库

| 模型         | 路径                                                    |
|:-----------|:------------------------------------------------------|
| Unet       | https://github.com/bubbliiiing/unet-pytorch           |
| PSPnet     | https://github.com/bubbliiiing/pspnet-pytorch         |
| deeplabv3+ | https://github.com/bubbliiiing/deeplabv3-plus-pytorch |
| hrnet      | https://github.com/bubbliiiing/hrnet-pytorch          |

### 性能情况

|   训练数据集   |                                                             权值文件名称                                                              |   测试数据集   | 输入图片大小  | mIOU  | 
|:---------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------:|:-------:|:-----:| 
| VOC12+SBD | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val12 | 512x512 | 72.59 | 
| VOC12+SBD |    [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth)    | VOC-Val12 | 512x512 | 76.95 | 
| VOC12+SBD |                  [deeplab_hgnetv2.pth](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)                   | VOC-Val12 | 512x512 | 78.83 |
| VOC12+SBD |                  [translab_hgnetv2.pth](https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/tag/v0.0.2-beta)                 | VOC-Val12 | 512x512 | 80.23 |

#### 目前该项目支持的主干网络有

MobileNetv2 MobileNetv3 XCeption HGNetv2(HGNet由百度开发，仪酷智能接入deeplab)，

YOLOv8(S和M尺寸，目前存在低mIOU的问题，不推荐)

#### 目前该项目支持的分割头有

官方Deeplabv3+的头（采用ASPP)

仪酷智能科技的TransLab头(采用AIFI Transformer) 

您可以自由的组合主干和分割头



### 所需环境

参看requirements.txt

### 文件下载

比较新的deeplab_HGNetv2由仪酷智能科技提供 [链接](http://dl.aiblockly.com:8145/pretrained-model/seg/deeplab_hgnetv2.pth)

```SHA256: D5DD6AB2556F87B8F03F12CCC14DCBEBADF01123003E1FBF3DB749D6477DBF8F```

训练所需的deeplab_mobilenetv2.pth和deeplab_xception.pth可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 提取码: qqq4

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk

### 训练步骤

#### a、训练voc数据集

1、`pip install git+https://gitee.com/yiku-ai/hgnetv2-deeplabv3` 安装
2、下载config.ini 根据实际情况修改  
3、命令行输入`siren.train -c config文件路径`。

4、~~实际你也可以`python -m yiku.train -c config.ini`~~
#### b、训练自己的数据集

1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在config.ini下面，选择自己要使用的主干模型和下采样因子,支持的模型在预测步骤中有描述。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
6、注意修改train.py的num_classes为分类个数+1。    
7、运行`siren.train -c config文件路径`即可开始训练。

### 导出步骤
1、命令行输入`siren.export -c config文件路径 -f onnx`。onnx位于配置文件的训练结果文件夹。
-f 参数支持 onnx openvino 和paddle 其中 还有--half 只要这个flag存在 openvino就是导出FP16精度的模型，
这在较新XeGPU上相比FP32有两倍的提升。所以参数 可以输入`siren.export -h`

2、~~运行`python -m yiku.export -c config文件路径 -f onnx`也可以导出~~。


### 预测步骤

#### a、使用预训练权重

1、根据下载的模型 修改config.ini，直接输入`siren.pred -c config.ini -i http://just.样例.link/怒O大伟出奇迹.jpg -m 你模型的路径.pth`

-i 参数可以是图片 可以是uri 可以是相机索引 可以是视频 可以是文件夹

--show 这个flag设置后就会将结果弹窗弹出。

-h 查看所有参数的帮助



#### b、使用自己训练的权重

1、按照训练步骤训练。    
2、使用训练的config即可，无需手动指定权重。



可完成预测。    
4、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。


### Reference

https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus  
https://github.com/bubbliiiing/deeplabv3-plus-pytorch  
https://github.com/ultralytics/ultralytics  
