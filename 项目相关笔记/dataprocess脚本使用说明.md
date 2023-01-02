## Dataprocess脚本使用说明

由于`mmsegmentation`框架的数据处理方法为读取`jpg`格式的`image`图片，读取`png`格式的`lable`图片，但是从`ArcGIS pro`中导出的文件为了之后项目的可扩展性，我们采取`image`和`lable`都是`tif`格式，该格式能够很好的保存遥感图片的特点，即像素值并非为`0-255`区间和除了`RGB`可见光波段外的其余波段信息。

本脚本包含了以下几个函数。

