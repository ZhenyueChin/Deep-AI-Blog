---
title: Convolutional Neural Networks 卷积神经网络 (1)
date: 2018-01-25 16:31:45
tags:
categories: 深度学习笔记
---

_本文作为Dr Andrew Ng(吴恩达)在[Coursera](https://www.coursera.org/learn/convolutional-neural-networks/)上面的公开课的学习笔记而存在_

[本文博客链接](https://zhenyuechin.github.io/2018/01/22/ng2017convolutional1/)

# 第一周 - Convolutional Neural Network 卷积神经网络基础

## Computer Vision 计算机视觉

在深度学习的帮助下, 计算机视觉有了突破性的进展. 简单来说, 计算机视觉是教会计算机如何更好地去模拟人类"看"图像或者视频. 比如, 人看到一张猫的图片能够知道"这是一张猫的图片". 我们想让计算机也获得这个能力. 

与深度学习结合之后, 有了突破性进展的计算机视觉领域包括自动驾驶, 人脸识别等. 从两个方面来看, 深度学习和计算机视觉的结合令人兴奋. 第一, 由于近年来有大量数据被创造以及计算机的计算速度的大幅度提高, 我们能够完成很多在之前不可能完成的任务. 我们有理由相信, 未来我们能够用深度学习做到更多不可思议的计算机视觉应用. 其次, 计算机视觉中的技术或许可以对其他领域造成很有益的启发. Dr Andrew Ng说他本人即使在语音识别的工作中, 收到了很多来自计算机视觉的技术的启发. 

总得来说, 深度学习主要帮助计算机视觉解决三种问题. 第一, 图像识别. 比如说下图是一只猫的图片, 我们怎么样才能够使计算机也能够识别出这是一只猫呢? 

![cat](https://zhenyuechin.github.io/images/coursera-cnn/a_cat.jpeg)[1]

第二, 对象检测(Object detection). 比如说下图包含一些汽车们和行人们, 我们怎么样使得计算机能够识别出图片的哪些区域对应汽车们, 哪些区域对应行人们呢? 

![object_detection](https://zhenyuechin.github.io/images/coursera-cnn/object_detection.jpg)[2]

第三, 艺术创作. 我本科期间的一个同学做了与此相关的工作, 他利用计算机创造一些绘画作品, 虽然我觉得他的计算机创造出来的"艺术品"和他本人创造出来的一样糟糕哈哈. 

细心的你可能注意到了, "嘿为什么我们需要卷积神经网络, 传统的深度神经网络不能解决这些问题吗"? 使用传统深度神经网络的一个问题是我们的输入太多了. 目前大家都喜欢"高清无码"图, 他们的分辨率很正常就能达到"1920X1080". 再加上这是彩图, 所以一共会有"1920X1080X3=6220800"个值输入到神经网络中. 这太大了. 

所以, 我们需要卷积操作来降低输入的值的数量. 至于具体怎么做, 请看下回分解. 

## Edge Detection Example 边缘检测

我本人对于边缘检测感到兴奋, 因为利用卷积操作来检测边缘是计算机模拟我们生物体操作的一个典型成功例子. 神经学家David Hubel和Torsten Wiesel在杀死了无数只猫之后, 发现动物在观察东西的时候, 是“卷积”的. 换句话说, 我们先看到物体的边缘, 然后把这些边缘组合起来看到更复杂的图像, 一层一层向上传播, 知道我们分析出了这个图像究竟是什么. 

卷积神经网络模拟了这个思想. 我们不再把整个图像作为输入值. 与此相对, 我们首先找出边缘, 然后把边缘作为输入. 这样我们就不会有过大的输入. 

那怎么样去找边缘呢? 我们仍然在模拟生物过程. 我们人类在看东西的时候是靠感光细胞. 如果两个相邻的感光细胞感受到的光的强度差别很大, 那么这两个感光细胞就知道边缘在他们上面. 我们使用"卷积"操作来模拟这个过程. 我将给出一个简单的例子. 

我们利用一个3X3的矩阵来模拟感光细胞. 卷积实际上是一个很简单的操作, 然而过程写出来却很困难. 

![convolution_operations](https://zhenyuechin.github.io/images/coursera-cnn/convolution_operations.png)[2]

恳请你们结合上面的绿色式子以及图像来想象一下卷积操作是怎么回事儿(怎么样找竖着的边缘). 请注意中间那个星号(*)不是乘号, 是卷积的意思. 

这儿还有个例子, 帮助你确定你理解的对不对. 

![vertical_edge_detection_examples](https://zhenyuechin.github.io/images/coursera-cnn/vertical_edge_detection_examples.png)[2]

中间那个矩阵就是感光细胞, 在卷积神经网络里面叫过滤器(filter). 左边一列全是1, 右边一列全是-1.这个值其实不重要, 只要他们互为相反数, 可以是2和-2, 也可以是3和-3. 请大家想象一下, 如果图像颜色一样的话, 那么卷积的结果将会是0. 如果两边颜色有差异, 那么差异越大, 卷积结果的绝对值也将越大. 

过滤器实际上是一个很好的名字, 把不需要的特征过滤走. 接下来或许你也会体会到这个名字的精妙. 

接下来这个动画或许能够进一步帮助你的理解. 在此, $$\begin{bmatrix}4 & 3 & 4 \\\\ 2 & 4 & 3 \\\\ 2 & 3 & 4 \end{bmatrix}$$ 为我们的过滤器. 

![conv_net_animation](https://zhenyuechin.github.io/images/coursera-cnn/conv_net_animation.gif)[3]

## More Edge Detection 更多边缘检测

不难想象检测水平的边缘的过滤器会是什么样子(水平的). 请参阅下图. 

![horizontal_edge_detection_examples](https://zhenyuechin.github.io/images/coursera-cnn/horizontal_edge_detection_examples.png)[2]

在神经网络中, 我们将不断改变过滤器中的值. 这将是卷积神经网络的主要学习过程. 

## Padding 衬垫

现在我们知道了, 如果你有一个\\(6 \times 6\\)的图像和一个\\(3 \times 3\\)的过滤器, 你将获得一个\\(4 \times 4\\)的卷积结果. 更加一般地, 当你有一个$N \times N$的图像和$f \times f$的过滤器时, 卷积的尺寸将是$(n-f+1) \times (n-f+1)$. 这很好理解, 卷积向右移动了$n-f$次, 但第一次不移动也占了1个格, 所以卷积之后的尺寸是$(n-f+1) \times (n-f+1)$. 

那么这样的话会有两个潜在的问题. 首先, 如果我们的卷积神经网络很深, 也就是说层数很多的话, 那么最后我们的图像尺寸会变为0. 我们就没有东西可处理了. 其次, 图片边边角角的像素被卷积的次数很少. 换句话说, 边边角角很少被使用. 我们会损失一些在边边角角的信息. 

为了解决这些问题, 我们在图像外围人为地加上一些像素层. 这些人为加入的层为衬垫(padding)格. 如下图所示, 对于一个原先尺寸为$6 \times 6$的图像, 假设我们加入了$p=1$个padding格, 那么我们现在的图像将会有$6+1+1=8$层. 卷积之后, 我们仍然能够有$6 \times 6$的尺寸. 不难想象, 一般地, 我们卷积的结果的尺寸将会如以下公式. 

![padding](https://zhenyuechin.github.io/images/coursera-cnn/padding.png)[2]

对于衬垫, 我们有一些术语. 

第一, 有效(valid): 也就是不加衬垫的卷积. 

第二, 相同(same): 也就是卷积之后的图像尺寸和卷积前相同. 具体来说, 我们想要: 

$$
\begin{equation}
n+2p-f+1=n \\\\
p=\frac{f-1}{2}
\end{equation}
$$

细心的你可能会说, 嘿如果过滤器的尺寸f是一个偶数, 那么我们的图像的尺寸不就是个小数了吗? 这有问题诶! 是的. 然而在卷积神经网络中, 我们约定俗称使用过滤器的尺寸f为奇数. 这样做还有另外一个好处, 我们的过滤器可以有一个中间位置, 这个位置有时候能包含更多的信息. 

## Strided Convolutions 跨步卷积

跨步卷积的思想实际上非常简单. 之前我们移动过滤器一个格子, 现在我们将此推广开来, 每次移动s个格子. 不难想象, 如下图所示, 卷积后的图像的尺寸将变成
$$
\begin{equation}
    \lfloor{\frac{n+2p-f}{s}+1}\rfloor \times \lfloor{\frac{n+2p-f}{s}+1}\rfloor
 \end{equation}
$$

![summary_of_conv](https://zhenyuechin.github.io/images/coursera-cnn/summary_of_conv.png)[2]

至于为什么取底数(floor)而不是取顶数(ceiling)也很好理解, 我们的过滤器不能飞出我们的图像之外呀. 换句话说, 我们的图像不能够多给我们的过滤器多走一步. 

神经网络中的卷积操作实际上和数学里面有所不同. 但我觉得这个不同并不重要. 请大家知道在神经网络里面卷积操作就是我们所讨论的样子就好啦. 

## Convolutions Over Volume 批量卷积

我们在小时候学过三原色的原理. 换句话说, 每一个图像可以想象成三种颜色的矩阵堆砌而成, 红, 绿, 蓝, 如下图所示. 

![rgb](https://zhenyuechin.github.io/images/coursera-cnn/images/rgb.png)[2]

因此, 对于每一个彩图, 我们可以使用三个不同的过滤器. 每一个过滤器对应不同的颜色. 然后, 我们把三个过滤器的卷积结果加起来作为最终的卷积结果. 换句话说, 我们将一个3D的输入彩图, 变成了一个2D的结果矩阵. 如下图所示, 左边的输入是3D的, 右边的输出是2D的. 

![convolutions_on_rgb_image](https://zhenyuechin.github.io/images/coursera-cnn/convolutions_on_rgb_image.png)[2]

之前我们讨论过, 竖直的过滤器和水平的过滤器是两种过滤器. 有时我们甚至需要45度的过滤器. 也就是说, 对于同一个输入, 我们需要不止一种过滤器. 比如, 我们可能同时有检测竖直边界的过滤器和检测水平边界的过滤器. 这两种过滤器各由三个过滤器组成, 用来检测红绿蓝色. 每一种检测不同边界的过滤器将会输出一个2D的图像. 如果我们有两种过滤器, 我们将会输出两个2D的图像. 如下图所示. 中间的为过滤器. 上面的三个过滤器可以对应检测竖直边界, 下面的三个可以对应检测水平边界. 

![multiple_layers](https://zhenyuechin.github.io/images/coursera-cnn/multiple_layers.png)[2]

## One Layer of a Convolutional Network 卷积网络的一层

前面说了很多关于卷积神经网络的知识, 接下来我们将给出卷积神经网络具体的一层. 

根据我们之前讨论的结果, 如果我们不考虑衬垫, 那么对于一个$6 \times 6$的图像, 将它卷积之后, 我们能够获得一个$4 \times 4$的图像. 我们接下来需要做的是加一个偏差(bias)在这个$4 \times 4$的图像. 换句话说, 我们给每一格加上bias. 同理, 我们也要对这个$4 \times 4$的新图像应用激活函数(activation function). 如果拿深度神经网络来比喻的话, 卷积操作就像之前的权重矩阵$W$一样. 输入$a$是输入图像. 偏差和之前并没有什么差别. 

Dr Andrew Ng提出了一个很好的练习, 我把它放在这里: 

如果你有10个过滤器, 每一个的尺寸是$3 \times 3 \times 3$. 那么我们有多少个参数呢? 

答案是280. 因为每一个过滤器有$3 \times 3 \times 3 + 1$个参数. 那个加一是因为每一个过滤器会有一个偏差. 

这个数字是很令人兴奋的! 因为对于图像问题来说, 特征数都是惊人的. 之前我们说过, 日常高清无码图会有几百万个特征. 然而, 这么多特征, 我们却只有不到300个参数要学习. 特征数比参数数多很多的事实有效地避免了过学习(overfitting). 这也是卷积神经网络的优点之一, 也就是没有那么多参数要学习. 

Dr Andrew Ng有一个很好的总结课件, 我将把它放在这里作为本小结的总结. 

![summary_of_notation](https://zhenyuechin.github.io/images/coursera-cnn/summary_of_notation.png)[2]

## A Simple Convolution Network Example 一个简单的卷积神经网络例子

之前我们说了卷积神经网络中的一层该怎么计算, 现在我们给出一个卷积神经网络完整的例子. 

如下图所示, 我们的输入图片是一个$39 \times 39 \times 3$的很小的图片. 第一层, 我们使用$3 \times 3$的10个过滤器, 跨步为1, 衬垫为0. 作为结果, 我们得到了10个$37 \times 37$的图像. 第二次, 我们使用$5 \times 5$的40个过滤器, 跨步为2, 衬垫为0. 最终, 我们得到了40个$7 \times 7$的图像.

紧接着, 我们把那40个$7 \times 7$的图像写成一个竖着的向量的形式, 作为深度神经网络的输入向量. 经过线性变化和激励函数变换之后, 我们最终给出分类结果(e.g. 输入的图像到底是不是一个猫的图像). 

![example_convnet](https://zhenyuechin.github.io/images/coursera-cnn/example_convnet.png)[2]

值得注意的是, 当我们的跨步值大于1时, 我们的结果图像的尺寸会下降快很多. 另外, 在深度增加时, 一般情况下, 是图像尺寸会减小, 然而图像的频道(channel)会增加. 频道是说有多少个输出的图像. 

## Pooling Layers 池化层

在卷积神经网络中, 我们经常使用池化层来增加我们的计算速度, 以及挑出来更加重要的特征和消除不太重要的特征, 使我们的网络更加健壮. 换句话说, 即使输入的图片质量不是很高, 我们还是可以得出正确的分类结果. 

我们首先来介绍什么是池化(pooling). 池化是一个很简单的概念. 我们以"最大池化"作为例子. 比如下图的$4 \times 4$的图像, 我们把它平均分割成4块区域, 每一块区域取最大值. 这样我们能够获得一个$2 \times 2$的区域. 在此, 我们相当于使用了一个f=2, s=2的过滤器. 

![pooling_layer_max_pooling](https://zhenyuechin.github.io/images/coursera-cnn/pooling_layer_max_pooling.png)[2]

这有点类似于人民代表大会制度. 人民代表大会制度是从每个镇选出最有影响力的人大代表, 然后在每个县选出最有影响力的镇人大代表, 逐层递增, 直到选出来全国人大代表. 我们的池化层也是这样, 我们在每一层选出来最有影响力的特征, 然后让它代表它所在的区域. 这样, 不太好的特征可以被好的特征"代表", 从而使整个网络并不会因为一些不好的特征影响. 值得注意的是, 池化层是没有参数的. 

我们给以个f=3, s=1的池化层作为一个例子来检查理解的正确性. 

![max_pooling_2](https://zhenyuechin.github.io/images/coursera-cnn/max_pooling_2.png)[2]

与最大池化相对的, 还有平均池化. 然而, 平均池化并不常用. 因为如果在一个区域存在不好的特征的话, 平均池化还是会引入这些不好的特征, 然而最大池化可以把这些不好的特征"和谐"掉. 

## Convolutional Neural Network Example 卷积神经网络例子

之前在没有池化层的时候, 我们给出了一个卷积神经网络的例子. 现在我们既有卷积层, 又有池化层. 我们将给出一个更加完善的例子. 

值得注意的是, 在卷积神经网络中, 我们一般只将有参数的层看成一个新的层. 换句话说, 我们不将池化层看作一个新的层. 

![cnn_example](https://zhenyuechin.github.io/images/coursera-cnn/cnn_example.png)[2]

## Why Convolutions? 为什么卷积？ 

使用卷积神经网络主要有两个好处: 第一, 共享参数; 第二, 稀疏连接. 

比如, 假设我们有一个输入图像, 这个图像的尺寸是$32 \times 32 \times 3 = 3072$. 经过6个过滤器的卷积之后, 我们获得了$28 \times 28 \times 6 = 4704$的结果. 如果使用普通完全连接(fully connected)深度神经网络, 这一步的计算涉及了一个$3072 \times 4704$的矩阵, 也就是这么多参数. 然而, 对于卷积神经网络, 如果$f=5$, 有6个过滤器, 我们只需要$6 \times (5 \times 5 + 1)$个参数(那个+1是因为偏差). 

然而, 为什么卷积神经网络只需要这么少数量的参数呢? 

第一是因为参数共享(parameter sharing). 对于卷积神经网络, 一个过滤器如果对于图片的一部分表现的很好, 那么它很有可能对于图片的另一部分也表现的很好. 比如说, 边缘检测过滤器对于左上角区域表现很好, 那么它很有可能对于右下角也表现的很好. 这样, 过滤器的尺寸不会因为输入图像的尺寸的增大而增大. 所以, 即使对于很大的图像, 我们并不需要很大的过滤器. 而过滤器是卷积神经网络主要参数来源. 

第二是因为稀疏连接. 这样, 结果图和输入图之间的关系就会大幅度减小. 这是因为卷积之后的每一个数值只和原图像的一个区域有关, 和其他区域都无关. 所以, 每个输出只和一部分输入有关, 而不是和整个输入有关. 这样, 输入和输出的关系就会大大降低. 神经网络学习主要是学习输入和输出之间的关系. 然而, 我们本来就没有很多关系要学习, 当然就不需要很多参数啦. 

也因此, 卷积神经网络很不容易发生过学习. 这是因为我们没有很多参数. 在神经网络中, 我们通过参数来记录学习过的东西. 过学习是因为我们记录的学习的东西太多了, 造成了"先入为主". 对于之后的判断, 之前的学习会影响之后的判断的准确性. 这可能是因为之前学习的很多特征在之后没有出现, 从而使得神经网络无法得出正确的结果. 但是, 我们本来就没有很多参数, 自然我们就不可能发生过学习. 

这也是为什么卷积神经网络可以解决"图像移动问题". 也就是说, 一个图像平移一段距离, 卷积神经网络仍能给出正确结果. 这是因为每一个过滤器只考虑一小块图像区域. 如果平移一段距离, 之前的过滤器仍能够走到之前的区域, 得到相同的卷积结果. 唯一不同的是这些过滤器要走稍微长一点距离. 

## References 参考文献

[1]"Grey and White Short Fur Cat · Free Stock Photo", Pexels.com, 2018. [Online]. Available: https://www.pexels.com/photo/grey-and-white-short-fur-cat-104827/. [Accessed: 25- Jan- 2018].

[2]"Convolutional Neural Networks | Coursera", Coursera, 2018. [Online]. Available: https://www.coursera.org/learn/convolutional-neural-networks/. [Accessed: 25- Jan- 2018].

[3]Saul Berardo, "What does the convolution step in a Convolutional Neural Network do?", Stats.stackexchange.com, 2018. [Online]. Available: https://stats.stackexchange.com/questions/116362/what-does-the-convolution-step-in-a-convolutional-neural-network-do. [Accessed: 25- Jan- 2018].

