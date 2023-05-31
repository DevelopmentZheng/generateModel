# stable diffusion学习内容系统表述

#  学习stable要学习哪一些内容

https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch

- 扩散模型的原理。 
- **使用UNet 模型** 的图像模型评分函数 
- **通过语境化词嵌入** 理解提示
- **让文本通过交叉注意力** 影响图像
- 通过添加**自动编码器提高效率** 
- 大规模训练

# ppt初步了解stable diffusion

stable diffusion的网络结构图

![截屏2023-05-25 下午4.12.56](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午4.12.56.png)

# 通过代码了解stable diffuion模型内部架构

https://colab.research.google.com/drive/1TvOlX2_l4pCBOKjDI672JcMm4q68sKrA?usp=sharing

自己添加注释以及说明版本。





# 通过大佬讲解了解了stable diffusion

https://jalammar.github.io/illustrated-stable-diffusion/

## 稳定扩散的主要组件

![截屏2023-05-24 上午10.07.25](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-24 上午10.07.25.png)

## 了解stable diffusion里面重要模块 diffusion

### 发生在哪里

![截屏2023-05-25 下午3.25.57](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.25.57.png)

### 把充满噪声的图片变成图片

![截屏2023-05-25 下午3.26.29](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.26.29.png)

#### 是一步一步的减掉噪声的

![截屏2023-05-25 下午3.27.22](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.27.22.png)



![截屏2023-05-25 下午3.27.56](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.27.56.png)

### 如何实现呢

#### 先构造数据集

构造出原始图片和添加噪声的，以及添加噪声后的图片

#### 通过数据集来训练unet模型实现预测噪声

![截屏2023-05-25 下午3.30.46](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.30.46.png)

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

 这个论文主要讲的就是这个方法。

## 压缩（潜在）数据而不是像素图像的扩散（更厉害了）

why

为速度更快

### 自动编码器使用其编码器将图像压缩到潜在空间

然后在进行扩散流程，

在潜在空间里面生成数据集，然后在训练。去噪模型unet。

## 添加文本编码器Clip，来影响图片

### 文本编码器影响很大

### 了解CLIP 是如何训练

### 将文本信息输入图像生成过程

把文本信息也输入噪声预测器里面，通过自注意力机制来实现。

![截屏2023-05-25 下午3.59.14](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-25 下午3.59.14.png)

# 带注释和简化的代码：[U-Net for Stable Diffusion (labml.ai)](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)

# 学习diffusion

# DDPM



# Score Diffusion Model分数扩散模型

b站视频

https://www.bilibili.com/video/BV1Dd4y1A7oz/?spm_id_from=333.788&vd_source=6d6126fdf98a0a7f2e284aa4d2066198

看到54分钟实在看不下去了，看一下作者博客

作者博客

https://yang-song.net/blog/2021/score/#perturbing-data-with-an-sde

作者代码

https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=GFuMaPov5HlV

## distribution

假设有 $未知p_{data}(x)数据中有一组独立同分布的数据\{\mathbf{x}_{i} \in \mathbb{R}^D \}_{i=0}^N $  D代表D维，N代表N个，R代表向量？

他的概率密度函数p(x)



## Score

what

概率密度函数p(x),我们规定 score 是$\nabla_\mathbf{x} \log p(\mathbf{x})$ 对概率密度取log，在求梯度

why

因为如果直接对概率密度建模的话，我们定义一个 $f_{\theta}(x)$ 网络模型，

然后不知道他们怎么求得出这个的 概率密度$ p_\theta(\textbf{x})=\frac{e^{-f_\theta(\textbf{x})}}{Z_\theta}$ ，（懂了看博客图片懂了，因为概率密度函数的积分必须是1，所以我们通过对$f_{\theta}$ 的改造添加$Z_{\theta}$ 称为归一化常数（没有变量） ，以及变成e的幂。让他的积分变成1.

 但是$Z_{\theta}$ 很难求，所以大佬在想把他消掉。

就想出一个分数，具体操作对密度函数 取log（log里面的除号可以变+号），然后在取梯度（由于没有变量x，所以梯度等于零就会被消掉）
$$
\textbf{s}_{\theta}(\textbf{x})=\nabla_{\textbf{x}}\log p_{\theta}(\textbf{x})=-\nabla_{\textbf{x}}f_{\theta}(\textbf{x})-\underbrace{\nabla_{\textbf{x}}\log Z_{\theta}}_{=0}=-\nabla_{\textbf{x}}f_{\theta}(\textbf{x}).
$$


##  Score-Based Models

然后我们创建一个模型score分数模型 来估计上面的分数

$S_{\theta}表示模型，输入\mathbb{R}^D->得到\mathbb{R}^D， $ 得到新的数据分数跟p(x)分数接近

 

why

直接对概率密度（分布）建模不好建，但是基于分数建模好建

基于分数的模型不需要被归一化，而且更容易设置参数。

举例一个以能量函数为概率密度

## 朗之万动力学采样

what

可以从概率密度p(x)中采样一个样本也服从概率密度p(x)，通过如下公式

$\overline{x}_{i+1}=\overline{x}_{i}+ \epsilon\nabla_\mathbf{x} \log p(\mathbf{x}) +\sqrt{\overline{x}_{i+1}} Z_{t}$ 

一个初始值 $\overline{x}_{0} 从 \pi(x)这个先验分布中采样出来的$, 加上分数函数,在加上一个初始值，经过给定一个步长 $\epsilon$  ,当多次迭代后，i变得无穷大，$\epsilon$  接近0时，获得的 $\overline{x}_{i+1}$ 就非常接近p(x)的分布，这就是朗之万采样



由朗之万采样，我们就可以采样一个我们想要得到的服从某个分布的数据了。当然前提是分数模型效果好

$S_{\theta} 接近 \epsilon\nabla_\mathbf{x} \log p(\mathbf{x}) $

why



## score match分数匹配

怎么让分数模型收敛呢

通过缩小loss，公式如下

$\mathbb{E}_{Pdata} [||S_{\theta}-\epsilon\nabla_\mathbf{x} \log p(\mathbf{x})||_{2}^{2}]$ 

不太懂为什么要求期望。

然后他可以通过雅可比矩阵来优化成如下公式

$\mathbb{E}_{Pdata(x)} [tr(\nabla_\mathbf{x} s_{\theta}(x))+0.5||s_{\theta}(x))||_{2}^{2}]$ 

但是雅克比矩阵有很难计算高维，所以用另外的方法，

why 

因为分数匹配可以在不知道p(x)的情况下来实现这个收敛。

缺点

分数匹配的方法，当数据分布比较稀疏的时候是分数函数会比较不准，那么后面的分数匹配也会不准，朗之万采样也会不准

## 去噪分数匹配

$\mathbb{E}_{q_{\sigma}(\overline{x}|x)Pdata(x)} [||S_{\theta}(\overline{x})-\epsilon\nabla_\mathbf{x} \log p(\mathbf{x})||_{2}^{2}]$  淘汰

我们把数据加噪，然后在计算他的分数模型loss，公式如下

![截屏2023-05-26 下午8.23.02](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午8.23.02.png)

所有的数据都变成了加噪后的数据，s模型是对加噪后的数据进行分数建模，

但是加噪后的数据是没加噪的数据还是有区别的，所以建模后的分数模型有一定的误差，不过如果加噪的数据比较小，那可以近似认为两个数据分布接近，然后就可以把分数模型带入朗之万公式里面。

why

既然数据分布比较稀疏，会不准，那么晚就加一点

缺点，

当分数密度比较低的时候，你加噪会对原始数据影响比较大，那么误差会比较多。

## 带噪声条件的分数网络 Noise conditional score networks （NCSN）

先定义一组等比数列。

![截屏2023-05-26 下午9.12.59](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午9.12.59.png)

不断变小的等比数列。

向数据分布Pdata(x)添加和这个等比数列相关的噪声，进行扰动。

扰动后的数据求积分用$Letq_{\sigma}(X)$ 表示

下面是进行扰动的公式，表示原始数据分布pdata(t)加上噪声的表示



![截屏2023-05-26 下午9.17.34](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午9.17.34.png)



然后我们希望训练的分数网络，能预测添加了不同噪声后的数据的分数。他就有两个输入，数据分数和噪声。

![截屏2023-05-26 下午9.22.11](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午9.22.11.png)

 我就把这模型网络称为噪声条件分数网络（NCSN）



why

上面的分数匹配和去噪分数匹配，效果都不好所以就想到这个方法

## 怎么建模NCSN

添加噪声后的数据我们用下面的$q_{\sigma}(\overline{X}|X) $表示，可以变成用高斯表示。

$q_{\sigma}(\overline{X}|X) = N(\overline{X}|X,\sigma^{2}I)$ 为什么又跟上面的不一样了

分数用如下表示

![截屏2023-05-26 下午9.40.23](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午9.40.23.png)

通过找到的数学的公式可以直接等价



![截屏2023-05-26 下午9.40.53](/Users/zhengyuxiang/Library/Application Support/typora-user-images/截屏2023-05-26 下午9.40.53.png)





然后建模loss

$\ell(\boldsymbol{\theta} ; \sigma) \triangleq \frac{1}{2} \mathbb{E}_{p_{\text {data }}(\mathbf{x})} \mathbb{E}_{\tilde{\mathbf{x}} \sim \mathcal{N}\left(\mathbf{x}, \sigma^2 I\right)}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\right\|_2^2\right]$ 



$\ell(\boldsymbol{\theta} ; \sigma)表示是什么条件下面的loss$  两个E表示的$P_{data}(x)下的数据分布，添加了后那个E的数据噪声$

后面那个就是做差，做最小二乘法， 

我们给训练数据不同的噪声添加后的分数，的损失一个不同的权重，来实现更好的效果。公式如下

$\mathcal{L}\left(\boldsymbol{\theta} ;\left\{\sigma_i\right\}_{i=1}^L\right) \triangleq \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_{i})\ell\left(\boldsymbol{\theta} ; \sigma_i\right)$,

然后实验发现  $\lambda(\sigma_{i}) =\sigma^{2} $ 结果很好，可以让公式跟$\sigma$ 无关

##  用退火朗之万(朗之万的变化)来从NCSN采样

$\overline{x}_{i+1}=\overline{x}_{i}+ \epsilon\nabla_\mathbf{x} \log p(\mathbf{x}) +\sqrt{\overline{x}_{i+1}} Z_{t}$ 





## 用扩散过程扰动数据

当我们将这种扩散过程反向用于样本生成时，分数将会出现

定义扩散过程如下

$\{\mathbf{x}(t) \in \mathbb{R}^d \}_{t=0}^T$  t是时间，d代表d个维度



扩散过程由随机微分方程控制
$$
\begin{align*}

d \mathbf{x} = \mathbf{f}(\mathbf{x}, t) d t + g(t) d \mathbf{w},

\end{align*}
$$





这个公式描述了一个以高斯分布为先验分布的后验分布。具体来说，$p(\mathbf{y})$是一个先验分布，$\mathcal{N}(\mathbf{x};\mathbf{y},\sigma_i^2I)$是一个以$\mathbf{y}$为均值，方差为$\sigma_i^2I$的高斯分布。那么$p_{\sigma_i}(\mathbf{x})$是在给定$\mathbf{x}$的情况下，对所有可能的$\mathbf{y}$进行积分，得到的后验分布。

换句话说，$p_{\sigma_i}(\mathbf{x})$表示当我们观察到$\mathbf{x}$时，我们对未知参数$\mathbf{y}$的不确定性的更新。这个更新是通过将先验分布$p(\mathbf{y})$与以$\mathbf{y}$为均值、方差为$\sigma_i^2I$的高斯分布进行卷积得到的。其中，$\sigma_i$是一个超参数，它控制了高斯分布的方差大小。当$\sigma_i$较小时，高斯分布的方差较小，表示我们对未知参数$\mathbf{y}$的估计已经比较确定，因此后验分布会更加集中。反之，当$\sigma_i$较大时，高斯分布的方差较大，表示我们对未知参数$\mathbf{y}$的估计不够确定，因此后验分布会更加分散。

总之，这个公式描述了一个贝叶斯推断的过程，通过先验分布和观测数据的结合，得到后验分布，从而对未知参数进行估计
