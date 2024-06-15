# Torch版的语义分割全家桶（大量模型一键训练与导出）

## 简介
yiku-seg 是一个由仪酷开发的语义分割工具，其中集成了大量的模型，同时使用主干网络和分割头分离的写法，使得在保证了傻瓜化训练的基础上,保留了模型配置的灵活性，其中包含了Translab等特色模型。在本文中将介绍如何使用yiku-seg训练一个模型

```pip install yiku_seg```

---


## Top News

<details> <summary>更新日志：增加数据增强开关</summary>

**`2024-06`** : **数据集支持多种文件格式**

下载源优化，这下国内也可以流程用了


**`2024-03`** : **增加数据增强开关 增加一堆模型**

~~OOOO天下第一~~

增加augmentation字段组 方便控制数据增强功能

新增了UNet+百度HGNetv2的杂交模型，可以在配置文件base字段里增加 arch=unet启用

**`2023-11`** : **增加Unet模型**

~~Unet伫立在大地之上~~

新增了UNet+百度HGNetv2的杂交模型，可以在配置文件base字段里增加 arch=unet启用

**`2023-11`** : **使用命令行工具训练**

~~”不行啊，每次都要改代码，感觉不如OO啊“~~

您还在为改python文件烦恼吗，您还在为乱哄哄的文件夹烦恼吗，赶快使用
`pip install git+https://gitee.com/yiku-ai/hgnetv2-deeplabv3` 安装吧，装了你不吃亏，装了你不上当。

更新了七彩VIP皮肤，不过不用担心，我已经给你充值好了，可以直接使用


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
</details>

### 模型文档

| [unet系列](./doc/unet.md) | [lab系列](./doc/lab.md)  | [pspnet系列](./doc/pspnet.md) | [segFormer系列](./doc/segformer.md)


### 所需环境

参看[requirements.txt](https://github.com/VIRobotics/hgnetv2-deeplabv3/blob/main/requirements.txt)

建议先安装pytorch


### 训练步骤

#### a、训练voc数据集

1、`pip install yiku_seg` 安装

2、VOC拓展数据集的百度网盘如下：
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk

3、下载config.ini 根据实际情况修改一般的，我们只需要去修改config.ini里面base字段的那些参数
```ini
[base]
frozen_batch-size=4
unfrozen_batch-size=2
frozen_epoch=100
unfrozen_epoch=50
fp16=true
dataset_path=VOCdevkit
save_path=logs
num_classes=21
backbone=hgnetv2l
image_size=512
header = transformer
```     
其中batchsize和你的显存大小有关path是路径相关的。num_classes一般是多少类 header和backbone与模型的结构有关，**建议打开config.ini里面有对参数的详细解释**


4、命令行输入`siren.train -c config文件路径`。

5、~~实际你也可以`python -m yiku.train -c config.ini`~~
#### b、训练自己的数据集

1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在config.ini下面，选择自己要使用的主干模型和下采样因子,支持的模型在预测步骤中有描述。下采样因子可以在8和16中选择。需要注意的是，预训练模型需要和主干模型相对应。   
6、注意修改train.py的num_classes为分类个数+1。    
7、如果您需要控制数据增强功能 请在config.ini里面添加如下字段组
```
[augmentation]
enable=true #总开关
jitter=0.3 # 尺寸抖动比率
flip=0.5 # 翻转图片比率
blur=0.25 # 模糊比率
hsv_jitter_enable=False #HSV抖动
```
8、运行`siren.train -c config文件路径`即可开始训练。

### 导出步骤
1、命令行输入`siren.export -c config文件路径 -f onnx`。onnx位于配置文件的训练结果文件夹。
-f 参数支持 onnx openvino 和paddle 其中 还有--half 只要这个flag存在 openvino就是导出FP16精度的模型，
这在较新XeGPU上相比FP32有两倍的提升。所以参数 可以输入`siren.export -h`

config,ini里面的base字段 加上`single_ch=true`可以启用单通道输入，~~虽然对推理性能没卵用，但有些人非得要这个模式~~

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
