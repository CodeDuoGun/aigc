# 基于transformer 的 比较式文本-图片预训练模型

1、结构
<image source="data/clip.png"> data/clip.png</image>

2、Patch+Position Embedding
    在VIT中，我们常设置这个卷积的卷积核大小为16x16，步长也为16x16，此时卷积就会每隔16个像素点进行一次特征提取，由于卷积核大小为16x16，两个图片区域的特征提取过程就不会有重叠。当我们输入的图片是224, 224, 3的时候，我们可以获得一个14, 14, 768的特征层。
    下一步就是将这个特征层组合成序列，组合的方式非常简单，就是将高宽维度进行平铺，14, 14, 768在高宽维度平铺后，获得一个196, 768的特征层。平铺完成后，我们会在图片序列中添加上Cls Token，该Token会作为一个单位的序列信息一起进行特征提取，图中的这个0*就是Cls Token，我们此时获得一个197, 768的特征层
    添加完成Cls Token后，再为所有特征添加上位置信息，这样网络才有区分不同区域的能力。添加方式其实也非常简单，我们生成一个197, 768的参数矩阵，这个参数矩阵是可训练的，把这个矩阵加上197, 768的特征层即可。

3、image encoder 图像编码器则采用CNN的ResNet或ViT架构
    transformer encoder （VIT vision transformer）
    获得shape为197, 768的序列信息后，将序列信息传入Transformer Encoder进行特征提取，这是Transformer特有的Multi-head Self-attention结构，通过自注意力机制，关注每个图片块的重要程度

4、text encoder 是一个基本的bert结构