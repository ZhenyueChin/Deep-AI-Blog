---
title: Convolutional Neural Networks 卷积神经网络 (1)
date: 2018-01-25 16:31:45
tags:
categories: 学习笔记
---

_本文作为Dr Andrew Ng(吴恩达)在[Coursera](https://www.coursera.org/learn/convolutional-neural-networks/)上面的公开课的学习笔记而存在_

# 第一周 - Convolutional Neural Network 卷积神经网络基础

## Computer Vision 计算机视觉

在深度学习的帮助下, 计算机视觉有了突破性的进展. 简单来说, 计算机视觉是教会计算机如何更好地去模拟人类"看"图像或者视频. 比如, 人看到一张猫的图片能够知道"这是一张猫的图片". 我们想让计算机也获得这个能力. 

与深度学习结合之后, 有了突破性进展的计算机视觉领域包括自动驾驶, 人脸识别等. 从两个方面来看, 深度学习和计算机视觉的结合令人兴奋. 第一, 由于近年来有大量数据被创造以及计算机的计算速度的大幅度提高, 我们能够完成很多在之前不可能完成的任务. 我们有理由相信, 未来我们能够用深度学习做到更多不可思议的计算机视觉应用. 其次, 计算机视觉中的技术或许可以对其他领域造成很有益的启发. Dr Andrew Ng说他本人即使在语音识别的工作中, 收到了很多来自计算机视觉的技术的启发. 

总得来说, 深度学习主要帮助计算机视觉解决三种问题. 第一, 图像识别. 比如说下图是一只猫的图片, 我们怎么样才能够使计算机也能够识别出这是一只猫呢? 

![cat](/images/a_cat.jpeg)[1]

第二, 对象检测(Object detection). 比如说下图包含一些汽车们和行人们, 我们怎么样使得计算机能够识别出图片的哪些区域对应汽车们, 哪些区域对应行人们呢? 

![object_detection](/images/object_detection.jpg)[2]

第三, 艺术创作. 我本科期间的一个同学做了与此相关的工作, 他利用计算机创造一些绘画作品, 虽然我觉得他的计算机创造出来的"艺术品"和他本人创造出来的一样糟糕哈哈. 

细心的你可能注意到了, "嘿为什么我们需要卷积神经网络, 传统的深度神经网络不能解决这些问题吗"? 使用传统深度神经网络的一个问题是我们的输入太多了. 目前大家都喜欢"高清无码"图, 他们的分辨率很正常就能达到"1920X1080". 再加上这是彩图, 所以一共会有"1920X1080X3=6220800"个值输入到神经网络中. 这太大了. 

所以, 我们需要卷积操作来降低输入的值的数量. 至于具体怎么做, 请看下回分解. 

## Edge Detection Example 边缘检测

我本人对于边缘检测感到兴奋, 因为利用卷积操作来检测边缘是计算机模拟我们生物体操作的一个典型成功例子. 神经学家David Hubel和Torsten Wiesel在杀死了无数只猫之后, 发现动物在观察东西的时候, 是“卷积”的. 换句话说, 我们先看到物体的边缘, 然后把这些边缘组合起来看到更复杂的图像, 一层一层向上传播, 知道我们分析出了这个图像究竟是什么. 

卷积神经网络模拟了这个思想. 我们不再把整个图像作为输入值. 与此相对, 我们首先找出边缘, 然后把边缘作为输入. 这样我们就不会有过大的输入. 

那怎么样去找边缘呢? 我们仍然在模拟生物过程. 我们人类在看东西的时候是靠感光细胞. 如果两个相邻的感光细胞感受到的光的强度差别很大, 那么这两个感光细胞就知道边缘在他们上面. 我们使用"卷积"操作来模拟这个过程. 我将给出一个简单的例子. 

我们利用一个3X3的矩阵来模拟感光细胞. 卷积实际上是一个很简单的操作, 然而过程写出来却很困难. 

恳请你们结合上面的绿色式子以及图像来想象一下卷积操作是怎么回事儿(怎么样找竖着的边缘). 请注意中间那个星号(*)不是乘号, 是卷积的意思. 

这儿还有个例子, 帮助你确定你理解的对不对. 

![vertical_edge_detection_examples](/images/vertical_edge_detection_examples.png)[2]

中间那个矩阵就是感光细胞, 在卷积神经网络里面叫过滤器(filter). 左边一列全是1, 右边一列全是-1.这个值其实不重要, 只要他们互为相反数, 可以是2和-2, 也可以是3和-3. 请大家想象一下, 如果图像颜色一样的话, 那么卷积的结果将会是0. 如果两边颜色有差异, 那么差异越大, 卷积结果的绝对值也将越大. 

![convolution_operations](/images/convolution_operations.png)[2]

接下来这个动画或许能够进一步帮助你的理解. 在此, $\begin{bmatrix}4 & 3 & 4 \\ 2 & 4 & 3 \\ 2 & 3 & 4 \end{bmatrix}$ 为我们的过滤器. 

![conv_net_animation](/images/conv_net_animation.gif)[3]

## More Edge Detection 更多边缘检测

不难想象检测水平的边缘的过滤器会是什么样子(水平的). 请参阅下图. 

![horizontal_edge_detection_examples](/images/horizontal_edge_detection_examples.png)[2]

在神经网络中, 我们将不断改变过滤器中的值. 这将是卷积神经网络的主要学习过程. 

## Padding 衬垫

现在我们知道了, 如果你有一个\\(6 \times 6\\)的图像和一个\\(3 \times 3\\)的过滤器, 你将获得一个\\(4 \times 4\\)的卷积结果. 更加一般地, 当你有一个$N \times N$的图像和$f \times f$的过滤器时, 卷积的尺寸将是$(n-f+1) \times (n-f+1)$. 

两个问题: 1. shrinking problem. 2. throwing away edge information.

when the neural networks are very deep, and you keep shrinking. Then you will run into problems. 

You can pad more pixels at the edge. 

Valid convolution: no padding. 
Same convolution: 

When f is odd, you can find a p so that the output size is the same as the input size. Normally we do not use an even filter size. (by convention)

odd filters have a central pixel

## 参考文献

[1]"Grey and White Short Fur Cat · Free Stock Photo", Pexels.com, 2018. [Online]. Available: https://www.pexels.com/photo/grey-and-white-short-fur-cat-104827/. [Accessed: 25- Jan- 2018].

[2]"Convolutional Neural Networks | Coursera", Coursera, 2018. [Online]. Available: https://www.coursera.org/learn/convolutional-neural-networks/. [Accessed: 25- Jan- 2018].

[3]Saul Berardo, "What does the convolution step in a Convolutional Neural Network do?", Stats.stackexchange.com, 2018. [Online]. Available: https://stats.stackexchange.com/questions/116362/what-does-the-convolution-step-in-a-convolutional-neural-network-do. [Accessed: 25- Jan- 2018].

