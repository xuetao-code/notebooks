[TOC]
### Overview
支持向量机(Support Vector Machine, SVM), 主要用于解决模式识别领域中的**数据分类**问题. 属于**有监督学习**的一种。
### 解决了什么问题
![4a8945157a9c6e04105d7c1168edd793.png](en-resource://database/5611:1)
如图a所示，红蓝两色的数据点是可以被一条直线分隔开的，在模式识别领域称为**线性可分问题**。而分割的直线不止一条，图b、图c分别给出了A、B两种解决方案，其中黑色的实线为分界线，数据称之为**决策面**，每个决策面对应了一个线性分类器。

SVM算法认为分类器A在性能上优于分类器B，其依据事A的分类间隔大于B。
> 分类间隔
> > 在保证决策面方向不变的前提下，两条虚线之间的垂直距离就是这个最优决策面对应的分类间隔。<u>具有最大间隔的决策面就是SVM要寻找的最优解</u>。
> 支持向量
> > 在最大间隔中，两侧虚线所穿过的样本点，就是SVM中的支持样本点，称为**支持向量**

### 线性SVM算法的数学建模
>优化问题通常有两个最基本的因素：
>    1. 目标函数：你希望什么东西的什么指标达到最好；
>    2. 优化对象：你期望通过改变哪些因素来是你的目标函数达到最优。

在线性SVM算法中，目标函数为分类间隔， 优化对象为决策面。

#### 决策面方程
1. 首先是初中学过的线性方程：
    $y=ax+b$
2. 然后把 $x, y$ 换个名字：$x_1, x_2$，得到：
    $x_2 = ax_1 + b$
    $ax_1 + (-1)x_2 + b = 0$
3. 把公式向量化：
    $[a,-1]\left[ \begin{array}{c}x_1\\x_2\end{array} \right] +b=0$
4. 转换一下:
    <img src="http://latex.codecogs.com/gif.latex?\boldsymbol{\omega}^T\boldsymbol{x}+\gamma=0" />
    其中<img src="http://latex.codecogs.com/gif.latex?\boldsymbol{\omega},\boldsymbol{x}" />表示这两个变量是个向量，$\boldsymbol{\omega}=[\omega_1,\omega_2]^T$, $\boldsymbol{x}=[x_1,x_2]^T$
    > 一般我们提到向量的时候，默认提到的是列向量，所以我们在方括号后面加上标T，表示转置

$\boldsymbol{\omega}=[\omega_1,\omega_2]^T$与直线$\boldsymbol{\omega}^T\boldsymbol{x}+\gamma=0$相互垂直，即$\boldsymbol{\omega}$控制了直线的方向。$\boldsymbol{\gamma}$则是直线的截距。

#### 分类间隔计算模型
![8d269fc6318cf39ebc14e93fb68a60c9.png](en-resource://database/5617:1)
首先我们得先复习下点到直线距离公式(2.6)：
```math
d = \frac{|\boldsymbol{\omega}^T\boldsymbol{x}+\gamma|}{||\boldsymbol{\omega}||}
```
> $||\boldsymbol{\omega}||$是向量$\boldsymbol{\omega}$的模
> $\boldsymbol{x}=[x_1,x_2]^T$就是支持向量样本点的坐标

所以我们追求目的找到了：追求d的最大化

#### 约束条件
1. 并不是所有的方向都存在能够实现100%正确分类的决策面，如何判断一条直线能够给所有样本正确分类？
2. 即使找到了正确的决策面方向，还要注意决策面的位置应该在间隔区域的中轴上，即截距$\gamma$也受到了决策面方向和样本点分布的约束
3. 即便找到了合适的方向和截距，距离公式中的$\boldsymbol{x}$必须是支持向量对应的样本点，如何找到对应的支持向量？

我们首先考虑一个据侧面能否将所有的样本都正确分类，我们为每个样本$\boldsymbol{x_i}$都加上一个类别标签$\boldsymbol{y_i}$:
```math
y_i = \left\{\begin{array}{ll}+1 & \textrm{for blue points}\\-1 & \textrm{for red points}\end{array}\right.
```
如果我们的决策面方程能够满足约束(1)，则有以下：
```math
\left\{\begin{array}{ll} \boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma>0 & \textrm{for~~} y_i=1\\\boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma<0 & \textrm{for~~} y_i=-1\end{array}\right.
```
如果我们要求高一点，假设决策面正好处于间隔区间的中轴线上，并且相对应的支持向量对应的样本点到决策面的距离为d，则得到：
```math

\left\{\begin{array}{ll} (\boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma)/||\boldsymbol{\omega}||\geq d & \forall~ y_i=1\\(\boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma)/||\boldsymbol{\omega}||\leq -d & \forall~y_i=-1\end{array}\right.
```
> $\forall~y_i$ 的意思是对于所有满足条件的y
我们对上面一组不等式两边除以d得到：
```math
\left\{\begin{array}{ll} \boldsymbol{\omega}_d^T\boldsymbol{x}_i+\gamma_d\geq 1 & \textrm{for~~} y_i=1\\\boldsymbol{\omega}_d^T\boldsymbol{x}_i+\gamma_d\leq -1 & \textrm{for~~} y_i=-1\end{array}\right.
```
其中：$\boldsymbol{\omega}_d = \frac{\boldsymbol{\omega}}{||\boldsymbol{\omega}||d},~~ \gamma_d = \frac{\gamma}{||\boldsymbol{\omega}||d}$

把$\boldsymbol{\omega_d}$和$\gamma_d$当作一条直线的方向矢量和截距，你会发现事情没有发生任何变化，因为直线$\boldsymbol{\omega_Tx}+y=0$其实就是一条直线。
所以我们可以得到一个结论：
> 对于存在分类间隔的两类样本，我们一定可以找到一些决策面，使其对于所有的样本点均满足下面的条件（2.12）：
```math
\left\{\begin{array}{ll} \boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma\geq 1 & \textrm{for~~} y_i=1\\\boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma\leq -1 & \textrm{for~~} y_i=-1\end{array}\right.
```

#### 线性SVM优化问题基本描述
在（2.12）中，当$\boldsymbol{x_i}$为决策面对应的支持向量的时，$\boldsymbol{\omega^Tx_i} + \gamma = 1 or -1$，所以我们可以替换掉(2.6)公式，得到(2.12)：
```math
d = \frac{|\boldsymbol{\omega^T x_i} + \gamma|}{||\omega||} = \frac{1}{||\omega||}
```
上述公式的意义是：支持向量样本点到决策面方程的距离就是$\frac{1}{||\omega||}$。
我们最初的任务是$W=2d$ 的最大化问题，这里可以转换为$||\boldsymbol{\omega}||$的最小化问题，即$||\boldsymbol{\omega}||$的最小化问题。
> 为了方便后期优化，对目标函数求导比较方便，我们可以给$||\omega||$加上平方和$\frac{1}{2}$系数（这不影响最后结果）

然后把类别标签$y_i$和不等式两边相乘，得到(2.13)：
```math
y_i(\boldsymbol{\omega^Tx_i} + \gamma) \geq 1~~~\forall~\boldsymbol{x_i}
```
这时，我们可以给出线性SVM的最优化问题的数学描述了：
```math
\begin{array}{l} \min_{\boldsymbol{\omega},\gamma}\frac{1}{2}||\boldsymbol{\omega}||^2\\ ~\\ \textrm{s. t.}~ ~y_i(\boldsymbol{\omega}^T\boldsymbol{x}_i+\gamma)\geq 1,~~i = 1,2,...,m \end{array}
```
>缩写 s. t. 表示"subject to"，即服从某某条件
