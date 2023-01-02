# `ArcGIS Pro`使用说明

​	本文档只是一个很入门的文档说明，仅仅针对本次项目，如果需要使用其他功能请自行百度。

## 软件工具的准备

首先通过百度网盘链接下载`zip`包

```
链接：https://pan.baidu.com/s/1IqXmeDoWIfK8XwT3gLyatA?pwd=Mb66 
提取码：Mb66 
```

下载完成后，按照`https://www.bilibili.com/read/cv19291680`所给出的教程进行破解安装。

## 将无锡市的数据导入到ArcGIS Pro

首先打开软件，新建`地图`工程。

![image-20221230180632029](C:\Users\anderien\AppData\Roaming\Typora\typora-user-images\image-20221230180632029.png)

​	工程新建后，可以看到左边有`内容`，讲初始话的两张地图删除，并且右键点击`内容`里的`地图`一栏，添加数据。

![image-20221230180734351](C:\Users\anderien\AppData\Roaming\Typora\typora-user-images\image-20221230180734351.png)

​	添加无锡市的矢量图斑，该文件是`shp`矢量文件，属于掩膜文件的一种，可以通过代码直接转换为带标签的语义分割`lable`，

​	添加无锡市的遥感影像，该文件是`tif`文件，属于栅格文件的一种，其包含多个波段，在未来的训练过程中我们会尝试加入多个波段而非传统的`RGB`可见光波段进行实验。

删除第四个红外波段只保留`RGB`波段需要用到`分析->栅格函数->提取波段`。



![image-20221230180804348](C:\Users\anderien\AppData\Roaming\Typora\typora-user-images\image-20221230180804348.png)

​	由于有一部分数据没有进行标注，所以此文档先略去标注的教程，我们通过矢量图来`resized`遥感图，这样就删除掉了多余的未标注地区，修改方法如下图所示。

![image-20230101202511818](E:\文档\寒假笔记\`ArcGIS Pro`使用说明.assets\image-20230101202511818.png)

添加好文件后我们需要做两件事，此处默认我们已经标注好了所有文件，其实有一部分矢量文件还有一小部分地图信息没有囊括，也就是说矢量文件和原图之间存在大小差异，而这部分差异正是我们需要进行人工标注的。

### 1.将矢量所有像素点的值转换为标签值

​	对于标签值得转换，这部分我没有找到简单的方法来进行转换。此处采取的是更换矢量文件对应的`dbf`文件来修改。

先找到`矢量图斑-无锡市/`目录下的`无锡市_qsdk_bg.dbf`,该文件实际上是一个小型数据库表，我们通过修改替换该文件来对矢量图中的像素点值进行修改，运行脚本文件中对应的`get_lable_from_dbf`脚本可以完成转换，转换完成后我们可以重新打开软件，查看矢量文件的属性表，右键点击`无锡市_qsdk_bg`选择属性表，如果属性表中有`Lable`属性则说明转换成功。

![image-20230101202206407](E:\mygithub\ArcGIS Pro使用笔记.assets\image-20230101202206407.png)

### 2.将矢量文件和遥感原图切割成能够满足深度学习的子图

​	具体分割方法请参照视频`https://www.bilibili.com/video/BV1mZ4y1X7sA/?spm_id_from=333.337.search-card.all.click`

以及`https://www.bilibili.com/video/BV1B44y1C7DA/?spm_id_from=333.337.search-card.all.click&vd_source=0e65ee2c9bd5062f4cbc3250f87d9217`

























