---
title: Lecture 1 Introduction to Reinforcement Learning 增强学习导论
date: 2018-01-28 21:07:07
categories: 增强学习笔记
---

_本文作为Prof David Silver的[增强学习公开课](https://youtu.be/2pWv7GOvuf0)笔记而存在_

[本文博客链接](https://zhenyuechin.github.io/2018/01/28/silver2015reinforcement1/)

# The Reinforcement Learning Problem 增强学习问题

## Many faces of RL 增强学习的多面

增强学习所体现的问题是"怎么样做出好的决定".还有一种情况是, "如果我们做出了乌龙决定, 我们怎么样能够避免下一次再做乌龙决定?". 实际上, 在很多其他学科中也有类似的问题. 比如说在经济学中, 我们有博弈论来分析类似的问题. 在神经科学中, 我们研究人脑是怎样做决定的. 人脑通过分析多巴胺来奖励好的决定. 这种机制实际上和增强学习有异曲同工之妙. 除此之外, 心理学研究其他动物怎么样做决定. 这些从一定程度上都可以说与增强学习有关. 

## Characteristics of RL 增强学习的特征

和其他机器学习方法比较来看, 增强学习有什么不同呢? 有以下几点: 

* 没有监督, 只有一个**奖励**信号. 
监督机器学习有一个类似于上帝的存在. 它能够告诉我们的主体(agent)每一步做得怎么样. 例如, "这一步能够给你打五分, 那一步能够给你打三分". 或者, "走这一步是最好的操作, 千万别走这一步, 它差极了". 在增强学习中, 我们没有这样类似于上帝的角色. 

* 反馈被延迟, 而不是及时给出. 
在监督学习中, 有一些在当下看似"很好"的操作可能会在一会儿显示出灾难性的结果. 然而, 这种灾难性的结果在当下并不能得到及时反馈. 

* 时间很重要 (时间是顺序的, 而不是i.i.d.(相同独立分布)的).
相同独立分布是说重复去做某件事情, 每件事情之间虽然可能会出现不同的结果, 但每种结果出现的可能性是相同的. 例如, 抛硬币. 虽然每次抛硬币可能会得出不同的结果, 但每一次都有相同的可能性是正面或反面(相同分布). 然而, 监督学习并不是这样. 监督学习不是猜抛硬币的结果, 你这次猜错了没关系, 下一次猜的结果和上一次没关系. 监督学习下一次的决定和上一次是有关系的. 比如说用监督学习训练会走路的机器人. 上一秒走到的地方直接影响下一秒做出的决定. 

* 主体的行为反过来会影响它接下来接收到的数据. 
与上一条类似, 在监督学习中, 不管你上一次表现如何, 我下一次该给你啥数据就还是啥数据, 与你上一次的表现无关. 但监督学习不一样, 比如一个选择去中国的机器人所接收到的环境信息和选择去日本的机器人大有不同. 

## Examples of Reinforcement Learning 增强学习的例子

* 阿尔法狗(AlphaGo). 
* 训练程序打Dota. 

# Inside An RL Agent 一个主体内部

## Reward 奖励

### Reward 奖励

奖励是一个**标量(scalar)**. 它反映了主体在某个时间$t$做得怎么样. 增强学习的主体的任务就是使自己积累的奖励最大化. 所以说, 奖励可以是负数. 

在这里, 我们有一个奖励假说. 你可以不同意他(我就不太同意哈哈, 如果你不同意, 欢迎用邮件与我交流: chin290956355@gmail.com). 

奖励假说: 所有的目标都可以被描述为最大化累计奖励的期望值 (All goals can be described by the maximisation of expected cumulative reward). 

### Sequential Decision Making 顺序决定作出 

顺序决定作出的目标是最大化将来的所有奖励. 每一个行为都可以有长期的影响. 奖励也可能被推迟. 所以, 为了最大化将来的奖励, 牺牲一些当前的蝇头小利可能是必要的. 所以不能太贪心. 

## Environment 环境

### Agent and Environment 主体与环境

在时间$t$时, 主体: 
* 执行行为$A_t$
* 接收观察$O_t$
* 接收奖励$R_t$

环境: 
* 执行行为$A_t$
* 放出观察$O_{t+1}$
* 放出奖励$R_{t+1}$

环境生成观察和奖励. 

![env_agent](https://zhenyuechin.github.io/images/silver2015reinforcement/env_agent.png)[1]

## State 状态

### History and State 历史与状态

历史(history)是一连串观察, 行为, 和奖励. 我们用$H$来代表历史. 在增强学习中, 环境是历史决定论. 换句话说, 过去的历史完全决定了下一刻的环境. 

$$
H_t = A_1, O_1, R_1, ..., A_t, O_t, R_t
$$

主体根据过去的历史做出一个行为. 换句话说, 主体将历史映射到一个行为上. 

然而, 虽然历史包含了大量的信息. 但因为历史太过于庞大. 如果我们使用历史来作出接下来的行为, 首先, 我们可能没有足够的空间来储存这么多信息; 其次, 我们可能没有足够的时间来给出下一个行为. 所以, 我们要对过去的历史进行一个总结. 这个总结就是状态(state). 

所以, **状态**是我们用来给出下一个行为所使用的工具. 换句话说, 我们把历史替换成了更精炼的历史. 

形式化地, 状态是历史的一个函数, 

$$
S_t = f(H_t)
$$ 

## Environment State 环境状态

我们用符号$S_{t}^{e}$来代表环境状态. 

环境在每个时刻有一个状态. 它能够总结环境的历史, 然后给予主体观测和奖励. 

实体大多数情况下并不是完全清楚环境状态. 比如, 我现在在咖啡馆里面坐着, 我并不知道中南海里面发生了什么事情. 另外, 并不是所有的环境状态都和实体有关. 比如银河系的某个黑洞现在如何和我关系不太大. 所以, 实体的动作并不取决于环境实体. 

## Agent State 主体状态

主体状态$S_{t}^{a}$代表了主体内部当前是什么样子. 它决定了主体接下来的行为. 它也是增强学习所使用的信息. 它决定了我们记录什么观察, 遗忘什么观察. 形式化地, 

$$
S_t^{a} = f(H_t)
$$ 

增强学习就是找出函数$f$是什么样子. 

## Information State (信息状态)

信息状态其实就是马尔科夫状态(Markov state). 形式化地, 

一个状态$S_t$是马尔科夫状态当且仅当

$$
P(S_{t+1} \| S_t) = P(S_{t+1} \| S_1, ..., S_t)
$$

换句话说, 当前状态只和之前的时间有关. 和之前的之前的时间的状态, 以及再之前都没有关系. 

## Fully Observable Environments 完全可观察环境

在这种情况下, 主体状态和环境状态是相同的. 换句话说, 主体就像上帝一样, 知晓环境的一切信息. 比如我现在在实验室, 知道中南海里面习主席在做什么. 形式化地, 

$$ 
O_t = S_t^a = S_t^e
$$

## Partially Observable Environments 部分可观察环境 

与完全可观察环境相对地, 我们还有部分可观察环境. 这是更常见的情形. 在这种情况下, 主体必须构建自己的状态. 主体有三种方式可以构建自己的状态: 

* 根据完整历史: $S_t^a = H_t$. 
* 相信环境变化是一个循环. 换句话说, 过段时间环境会重复一遍: $S_t^a = (P[S_t^e = S^1], ..., P[S_t^e = S^n])$. 
* 递归神经网络. 以不变应万变: $S_t^a = \sigma(S_{t-1}^{a} W_{s} + O_{t} W_{o})$. 

## Major components of an RL agent 增强学习主体的主要组成部分

一个主体可能包含以下成分. 请注意, 一个主体不一定包含以下所有成分. 

政策(policy): 主体的行为函数. 
价值函数(value function): 每一个状态 和/或 行为 有多好. 
模型(model): 主体对于环境的表示. 主体通过构建模型来认识环境.  

## Policy 政策

政策是从状态到行为的一个映射. 有两种政策. 

* 确定性政策: 一个状态只对应一个行为. 形式化地, $a = \pi(s)$
* 随机性政策: 一个状态在不同时刻可以对应不同的行为. 形式化地, $\pi(a | s) = P(A_t = a | S_t = s)$. 

## Value Function 价值函数

价值函数是对未来会获得的奖励的**预测**. 如果我们的主体是基于价值的, 那么我们并不需要给它定义政策. 它只需要计算一下每一个行为的价值, 然后选价值最好的就好了. 没必要明确告诉它"这样做". 

## Model 模型

模型预测环境接下来将如何变化. 我们用$P$来预测下一个状态, $R$来预测下一个奖励. 形式化地, 

$$
P_{ss'}^a =P(S_{t+1} = s' | S_t=s, A_t=a)
R_s^a =E(R_{t+1} | S_t =s, A_t = a]
$$

对于增强学习, 我们可以使用模型, 也可以不使用模型. 

# Problems within Reinforcement Learning 增强学习的问题

## Learning and Planning 学习和规划

在作出一系列决定问题中, 我们主要有两类问题: 

* 增强学习: 
- 环境一开始是未知的
- 主体与环境相互接触
- 主体改进它的政策

* 规划
- 环境的模型是已知的
- 主体不与环境接触, 只对模型进行计算
- 主体改进它的政策

## Exploration and Exploitation 探索与利用

增强学习就像"试错"学习. 

探索是发现关于环境的更多信息. 利用是利用已知的环境信息来最大化奖励. 探索和奖励之间的平衡非常重要.

比如, 每次我们出去吃饭的时候, 我们可以去那些我们已经知道食物好吃的饭馆, 这是利用; 我们当然也可以试一下新饭馆, 这是探索. 我们探索的饭馆可能更好吃, 也可能极难吃. 

# Future Work 

The ideas behind exploration and exploitation are similar to the ideas behind evolution strategies. For evolution strategies, they are exploring by mutating the original materials. However, how do they do the exploitation is unknown. Therefore, it will be interesting to figure out whether we can combine evolution strategies and reinforcement learning together. 

# References 参考文献

[1]"Introduction to Reinforcement Learning", David Silver, 2018. [Online]. Available: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/intro_RL.pdf. [Accessed: 29- Jan- 2018].












