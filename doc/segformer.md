
### 配置文件

```ini
[base]
arch = segformer
```


### 性能情况

| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mIOU | 
| :-----: | :-----: | :------: | :------: | :------: | 
| VOC12+SBD | [segformer_b0_weights_voc.pth](https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_weights_voc.pth) | VOC-Val12 | 512x512 | 73.34 | 
| VOC12+SBD | [segformer_b1_weights_voc.pth](https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_weights_voc.pth) | VOC-Val12 | 512x512 | 76.80 | 
| VOC12+SBD | [segformer_b2_weights_voc.pth](https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_weights_voc.pth) | VOC-Val12 | 512x512 | 80.38 | 
#### 目前该Arch支持的主干网络有

MobileNetv2 MobileNetv3 XCeption HGNetv2(HGNet由百度开发，仪酷智能接入deeplab)，

YOLOv8(S和M尺寸，目前存在低mIOU的问题，不推荐)

```ini
[base]
backbone = b3
```

| backbone |
|:--------:|
|    b0    |
|    b1    |
|    b2    |
|    b3    |
|    b4    |
|    b5    |

#### 目前该Arch支持的分割头有

此字段在此arch下无意义




### 文件下载

训练所需的权值可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1tH4wdGnACtIuGOoXb0_rAw    
提取码: tyjr    
VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk