---
title: Natural Evolution Strategies (自然进化策略)
date: 2018-01-22 18:00:03
tags:
comments: true
categories: 论文评论
---
_原作者: Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, Jurgen Schmidhuber_

[本文博客链接](https://zhenyuechin.github.io/2018/01/22/wirestra2014natural/)
[原论文链接](https://arxiv.org/abs/1209.5853)

Natural Evolution Strategies (NES), 以下翻译为自然选择策略, 是一种黑箱式优化算法. 所谓黑箱式优化算法, 是说我们只需要告诉计算机"什么解法是好的", "什么解法是坏的". 具体来说, 我们需要有一个函数, 该函数能够针对一个解法, 返回一个该解法"多好"的程度的值. 除此之外, 我们不需要操心怎么样找到好的解法. 这一切交给计算机做就好. 

目前主流的黑箱优化算法包括神经网络, 进化算法等. 传统算法难以解决的问题, 比如旅行推销员问题(Travelling salesman problem), 在这些新式算法的协助下, 得到了相当令人满意的答案. 

Wirestra等人提出了将进化算法和神经网络中的梯度下降思路结合在一起的想法. 传统的进化算法包含突变和重组这两个步骤. 我们通过这两个步骤, 期待找到更好的解法. 然而, 突变和重组是完全随机的. 多数情况下, 他们会导致和当前解法相比更差的解法. 因此, 我们想引入梯度下降(gradient descent)或梯度上升(gradient ascent)的思想, 从而使得突变总是能够朝着更好的解法迈进. 

换句话说, 我们用梯度下降替代了突变和重组步骤. 

P.S. 接下来我们会去实现NES. 并且我们会搞明白怎么用微信公众号打latex公式...

===一些好玩儿的想法

The (environmental) selection in evolution strategies is deterministic and only based on the fitness rankings, not on the actual fitness values. The resulting algorithm is therefore invariant with respect to monotonic transformations of the objective function. 哪怕在fitness function转化完成之后, 你大爷永远是你大爷(单调性). 

(1 + 1)-ES 子代只有可能青出于蓝而胜于蓝

\begin{align\*} 
J(\theta )=\operatorname {E}_{\theta }[f(x)]=\int f(x) \pi (x|\theta ) dx
\end{align\*}

这简直就是格差社会啊, 无视表现的差的个体. 用表现的好的个体统领总群的走向. 因为我们的目的是给予较大的\\( f(x) \\)最大的\\( \pi(x|\theta ) \\). 换句话说, 我们想找到一个\\( \theta \\), 使得较大的\\( f(x) \\)进步较大. 

:eyes: Ask Prof Bob McKay on how could the log transformation work. 

然而文章里面说这种search gradient有问题, 如果

:eyes: Ask Prof Bob McKay on the idea of natural search gradient. Ask h

There are two advantages of natural search over plain search gradient: 
* the gradient direction is independent of the parameterization of the search distribution. 
* the updates magnitudes are automatically adjusted based on uncertainty, in turn speeding convergence on plateaus and ridges.

I don't understand how can the formulae in the natural search gradient to facilitate the two properties above. However, 如果在一个搜索过程中, 搜索的进步与当前的变量是独立的, 那么的确我们可以期待持续进步. 

While evolution strategies have shown to be effective at black-box optimization, analyzing the actual dynamics of the procedure turns out to be difficult, the considerable efforts of various researchers notwithstanding. 
目前我们不太理解进化算法中所发生的动态过程. 

:eyes: Ask Prof Bob McKay for what is "parameters of density". 





