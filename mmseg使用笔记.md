## `mmsegmentation`框架使用说明

​	该框架为openmmlab开发，实现了许多语义分割方面的经典网络，虽然运行的结果可能比不上原论文，但是代码流程清晰，很适合进行项目开发和对比实验，本文档将以广西的数据集进行举例，以此来讲述如何用自定义数据集来进行网络训练。

​	本文档仅仅说明了如何使用自定义数据集运行`mmsegmentataion`框架，如果需要了解整个框架的处理流程等，请参考官方文档。`https://mmsegmentation.readthedocs.io/zh_CN/latest/train.html`

### 框架的安装与环境配置

​    源码的下载地址为`https://github.com/openmmlab/mmsegmentation`，遵循`Readme`文档中的官方教程来完成环境配置，如果在环境配置中存在一些问题，请在官方`issues`中尝试查看解决方案。

### 自定义数据集

​	首先我们有数据集`GX_data`，数据集结构如下所示

```
GX_data
  |---images
  	|---training
  		|---00000000000.png
  		|---00000000008.png
  			......
  	|---validation
  	|---test
  |---annotations
  	|---training
  		|---00000000000.png
  		|---00000000008.png
  			......
  	|---validation
  	|---test
```

其中`training`包括了`4121`张图片，`validation`包括了`1373`张图片，`test`为空集。

数据集准备好后我们需要修改项目框架中的几个文件来注册我们的自定义数据集。

#### 1.在`mmseg`库代码中添加`gx.py`数据读取文件

​	在`{yourdir}/mmseg/datasets`目录下新建`gx.py`文件，内容为

```python
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GXDataset(CustomDataset):
    CLASSES = ('background', 'cultivated_land', 'garden_meadow', 'construction_land', 'transportation_land', 'waters',
               'other_land', 'other_woodland')

    PALETTE = [[0, 0, 0], [0, 128, 128], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [128, 128, 128]]

    def __init__(self, **kwargs):
        super(WuxiDataset, self).__init__(
            img_suffix='', seg_map_suffix='', **kwargs)
        '''
        	注意这里后缀可能为空，也可能为数据集的图片后缀.png，如果报错为sample = 0，请尝试修改此处
        '''
        assert osp.exists(self.img_dir)

```

并且在`{yourdir}/mmseg/datasets/__init__.py`中的`__all__`列表末尾加入`GXDataset`

#### 2.修改调色板文件

​	在`{yourdir}/mmseg/core/evaluation/class_names.py`文件中，添加以下代码：

```python
def gx_classes():
    return ['background', 'cultivated_land', 'garden_meadow', 'construction_land', 'transportation_land', 'waters',
            'other_land', 'other_woodland']


def gx_palette():
    return [[0, 0, 0], [0, 128, 128], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [128, 128, 128]]
```

并且在`dataset_aliases`字典中添加`'gx': ['gx']`

#### 3.修改`configs`中的数据集文件

​	在`{yourdir}/configs/base/datasets`目录下新建`gx.py`,并添加以下内容，其中`data_root`修改为`GX_data`存储目录。

```python
# dataset settings
dataset_type = 'GXDataset'
# data_root = 'data/VOCdevkit/VOC2012'
data_root = '../data/Gx_data'	# 数据集存储目录
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,	# batchsize
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

```

### 修改网络配置文件

#### 1.配置网络文件

​	下面以`ccnet`举例，先进入`{yourdir}/configs/ccnet/`目录，新建一个我们希望用到的网络配置文件，例如`ccnet_r50-d8_512x512_160k_gx.py`，其中`r50`代表骨干网络采用`ResNet-50`；`d8`代表深度为`8`,一般不改变；`512x512`代表输入的图像尺寸为`512`；`160k`代表训练迭代为`160000`次，关于训练迭代次数和`batch size`、`sample_per_gpu`以及`epoch`之间的关系请自行百度。

​	在新建的`ccnet_r50-d8_512x512_160k_gx.py`文件中添加以下内容：

```python
_base_ = [
    '../_base_/models/ccnet_r50-d8.py',	# 已经定义好的网络，不需要进行改变
    '../_base_/datasets/gx.py',	# 自定义数据集
    '../_base_/default_runtime.py',		# 日志设置文件
    '../_base_/schedules/schedule_160k.py'	# 运行的部分超参数
]
model = dict(
    decode_head=dict(num_classes=8), auxiliary_head=dict(num_classes=8))	# 因为GX_data有8个类别，所以将预测数改为8
	"""
		实际上这里能修改网络的许多参数，相当能够重写../_base_/models/ccnet_r50-d8.py的所有参数，因此如果我们需要添加不同的loss或者使用辅助分类头，
		都能够在此处进行定义，具体方法参考../configs/下的文件定义方式以及官方文档。
	"""

```

​	如果需要更换不同的网络，则按照相同的方法在`configs/{select_modul}`下新建上述文件，并且内容参考同目录下的其余文件配置。

### 运行项目

#### 1.在pycharm中运行

​	在`{yourdir}/tool/train.py`文件中主要修改两个参数

##### 修改`--config`参数

![image-20230101193033846](E:\文档\寒假笔记\mmseg使用说明.assets\image-20230101193033846.png)

​	其中`default`设置成上面所提到的网络配置文件，注意`congfig`前有`--`，如果不添加将会报错。

##### 修改`--work-dir`参数

![image-20230101193214744](E:\文档\寒假笔记\mmseg使用说明.assets\image-20230101193214744.png)

该参数为最终模型日志和模型输出文件夹，如果文件夹不存在则会新建。`train.py`中的其他参数请参考对应的`help`说明，如果需要设置其流程遵循上述。

​	关于`tool`文件加中的其他脚本的功能以及运行方法都与`train.py`运行类似，不再赘述。

#### 2.在命令行中运行

​	如上所示，将执行目录切换为`{yourdir}/tool`，运行

```shell
python train.py --config '../configs/ccnet/cent_r101-d8_512x512_160k_gx.py' --work_dir '../gx/ccnet_r101_160k'
```

#### 3.采用`ViT`等`Transformer`网络

​	与基于`CNN`的网络不同，基于`Transformer`的网络不能够直接运行，需要预先下载与训练权重，再根据所下权重利用`{yourdir}/tools/model_converters`下的脚本进行转换，此处以`ViT`为例子。

​	根据`{yourdir}/configs/vit/README.md`文件的提示，先下载预训练权重。

![image-20230101195448516](E:\文档\寒假笔记\mmseg使用说明.assets\image-20230101195448516.png)

值得注意的是，有一些预训练权重的`url`是藏在代码里的。

![image-20230101195540211](E:\文档\寒假笔记\mmseg使用说明.assets\image-20230101195540211.png)

​	下载好后运行`{yourdir}tools/model_converters/vit2mmseg.py`文件进行转换，之后再修改`train.py`中的`--load-from`等参数进行训练。





### 运行效果

​	在运行成功后会打印出相对应的日志信息，包括运行时间、迭代数、学习率、剩余时间、两次日志输出之间的运行时间、数据处理时间、内存和`loss`等

![sunlogin_20230101194048](C:\Users\anderien\Documents\Sunlogin Files\sunlogin_20230101194048.bmp)

根据`{yourdir}/configs/_base_/schedules/schedule_160k.py`的设置，会输出对应的精度和权重文件。

```python
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)	# 优化器
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)	# 学习率策略
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)	# 总迭代数
checkpoint_config = dict(by_epoch=False, interval=16000)	# 每16000次迭代保存一次权重文件
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)	# 每16000次迭代计算一次mIoU,随日志文件一同输出到终端，并不会保存

```

最终在文件目录中能够找到日志和权重输出文件，此处以`upernet`举例，得到权重文件可以通过运行`test.py`文件进行精度评估。

![image-20230101194758221](E:\文档\寒假笔记\mmseg使用说明.assets\image-20230101194758221.png)

