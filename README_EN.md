##  All in One Tools for Semantic Segmentation with pyTorch 

---


## Top News

<details> <summary>Update Log： Add HGNetv2-Unetmodel</summary>

**`2023-11`** : **Add Unet Model**

~~Unet伫立在大地之上~~

 Add UNet+Baidu's HGNetv2 mixed model，Now you can  add  arch=unet in config.ini's base segment to enable it

**`2023-11`** : **Use CLI tool to train**

~~”不行啊，每次都要改代码，感觉不如OO啊“~~

you can use the cmd 
`pip install git+https://gitee.com/yiku-ai/hgnetv2-deeplabv3` to install 

<font color=red>Woo</font><font color=yellow>,</font><font color=green>Such</font> <font color=aqua>color</font>,
<font color=blue>So</font> <font color=DarkViolet>delicious</font>

I use rich-python instead of tqdm


**`2023-09`** : **Add TransLab Header，Use header=transformer to switch**

TransLab is developed by VIRobotics,Inc， We replace DeepLabv3's ASPP module to AIFI module,which is based on Transformer

~~玩Transformer玩的~~

**`2023-08`**:**Add a lot of Backbone to this repo**

issue and PR with bug report is welcome

VIRobotics's Industrial AI toolkit for LabView Now support all model in this repo

this repo is forked from https://github.com/bubbliiiing/deeplabv3-plus-pytorch

______
From orgin repo

**`2022-04`**:**支持多GPU训练。**

**`2022-03`**:**进行大幅度更新、支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/deeplabv3-plus-pytorch/tree/bilibili

**`2020-08`**:**创建仓库、支持多backbone、支持数据miou评估、标注数据处理、大量注释等。**
</details>

### Model 

[unet series](./doc/unet.md)

[lab series](./doc/lab.md)



### Enviroment

read  requirements.txt



### How to train

#### a、Train the voc Segment Dataset

1、`pip install git+https://gitee.com/yiku-ai/hgnetv2-deeplabv3` 
2、download config.ini and modify it 
3、input `siren.train -c config_path` to run

4、~~In fact`python -m yiku.train -c config.ini` is also ok~~
#### b、train datasets you own

1、We use VOC format to train。  
2、Put label file uder VOCdevkit/VOC2007/SegmentationClass folder    
3、Put image under VOCdevkit/VOC2007/JPEGImages。    
4、run voc_annotation.py to generate txt file。    
5、read [model section](#Model) and modify config.ini。   
6、Notice:num_classes in train.py shoud be num class in fact+1 (background)。    
7、run `siren.train -c config文件路径`to start。

### export
1、Input `siren.export -c config_path -f onnx`。result will generate in the folder recode in config。
-f support value in  onnx openvino 和paddle ,Arg --half is a store_true arg to enable fp16 model.
If you need help input `siren.export -h`

2、~~run`python -m yiku.export -c config文件路径 -f onnx`is also ok~~。


### predict

#### a、Use pretrained

1、download model  modify config.ini，input`siren.pred -c config.ini -i http://just.样例.link/怒O大伟出奇迹.jpg -m 你模型的路径.pth`

-i can be a picture a URI a video a dir or a CameraIndex

--show set this flag to pop out video playback

-h get help



#### b、Use trained by self

1、train。    
2、use config which used in train,no modify needed。



### Reference

https://github.com/ggyyzm/pytorch_segmentation  
https://github.com/bonlime/keras-deeplab-v3-plus  
https://github.com/bubbliiiing/deeplabv3-plus-pytorch  
https://github.com/ultralytics/ultralytics  
