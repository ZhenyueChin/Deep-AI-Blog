---
title: Lecture 2 Markov Decision Process 马可夫决策过程
date: 2018-01-29 19:15:29 
tags:
---

_本文作为Prof David Silver的[增强学习公开课](https://youtu.be/lfHX2hHRMVQ)笔记而存在_

[本文博客链接](https://zhenyuechin.github.io/2018/01/28/silver2015reinforcement2/)

# Markov Processes

## Introduction 

### Introduction to MDPs 

Markov Decision Processes formally describe an environment for reinforcement learning. 

This environment, we want some descriptions about this environment. 

The environment is fully observable. 

All the relevant information is presented to the agent. 

Almost all RL problems can be formalized as MDPs. 

optimal control primiarily deals with continuous MDPs 

any partial observable problems can be converted into MDPs 

## Markov Properties 

### State Transition Matrix 

### Markov Chains 

Markov Chain是finite states. 

A terminal state doesn't need any special machinery. You can just think it as a self-loop.  

A sample is a sequence of states 

transition matrix 看起来很稀疏啊. 

这就是之前学过的那个概率矩阵. 

# Markov Reward Processes

## Markov Reward Process (MRP)

Immediately reward. 

## Return 

The discount $\gamma \in [0, 1]$ is the present value of future rewards. It measures how much we care about the future reward. 

We only care about the immediate reward that we are going to get (That's all about RL). 

# Markov Decision Processes

# Extensions to MDPs