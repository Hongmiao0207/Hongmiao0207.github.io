# Machine Learning Notes 05


## 6. 支持向量机

它是一个几何角度建模模型，对于线性可分数据集，支持向量机就是找距离正负样本都最远的超平面。相比于感知机（它就是找到一个超平面能将正负分开就行，解不唯一），其解是唯一的，且不偏不倚，泛化性能更好。

### 6.0 预备知识

#### 超平面

{{<keepit>}}
一维超平面是点，三位超平面是个面，n维空间的超平面（$\omega^Tx+b=0$，其中$\omega, \ x \in{R^n}$）

* 超平面不唯一
* 法向量$\omega$和位移项b确定唯一一个超平面
* 法向量$\omega$垂直于超平面（缩放$\omega,b$时，若缩放倍数为负数会改变法向量方向）
* 法向量$\omega$指向的那一半空间为正空间（正空间的点代入方程是大于0），另一半为负空间（负空间点代入小于0）
* 任意点x到超平面的距离公式：
$$r=\frac{|\omega^Tx+b|}{||\omega||}$$

证明点到超平面的距离公式

证明：对于任意一点$x_0=(x^0_1,x^0_2,...,x^0_n)^T$，设其在超平面$\omega^Tx+b=0$上的投影点$x_1=(x^1_1,x^1_2,...,x^1_n)^T$，则$\omega^Tx_1+b=0$，且向量$\overrightarrow{x_1x_0}$与法向量$\omega$平行，因此

向量点乘：$a \cdot b = ||a|| \cdot cos\theta \cdot ||b||$ (向量a模长乘上cos两个的夹角乘上b的模长)
$$|\omega \overrightarrow{x_1x_0}| = ||\omega|| \ cos\pi \ ||\overrightarrow{x_1x_0}|| = ||\omega|| \ ||\overrightarrow{x_1x_0}|| = ||\omega|| r$$

$\overrightarrow{x_1x_0}$ 就是点到超平面的距离，记作r
$$\omega \overrightarrow{x_1x_0} = \omega_1(x^0_1-x^1_1)+\omega_2(x^0_2-x^1_2)+...+\omega_n(x^0_n-x^1_n)$$

$$=\omega_1x^0_1+\omega_2x^0_2+...+\omega_nx^0_n\omega_1x^1_1+\omega_2x^1_2+...+\omega_nx^1_n)$$

$$=\omega^Tx_0-x^Tx_1$$

其中，$\omega^Tx_0$等于证明假设中 $\omega^Tx_1$，$-x^Tx_1$ 等于 $\omega^Tx_1+b=0$ 的 b

$$=\omega^Tx_0+b$$

$$|\omega^Tx_0+b|=||\omega|| r \Rightarrow r = \frac{|\omega^Tx+b|}{||\omega||}$$
{{</keepit>}}

#### 几何间隔

{{<keepit>}}
对于给定的数据集X和超平面 $\omega^Tx+b=0$，定义数据集X中任意一个**样本点**$(x_i,y_i), \ y_i \in{\begin{Bmatrix} -1,1 \end{Bmatrix}}, \ i=1,2,...,m$ 关于超平面的几何间隔为（其中y取值为1，-1方便计算，也可以取值1，0都行）

$$\gamma_i=\frac{y_i(\omega^Tx_i+b)}{||\omega||}$$

正确分类时：$\gamma_i >0$，几何间隔此时等价于点到超平面距离。

没正确分类时，$\gamma_i <0$。

对于给定的数据集X和超平面，定义**数据集**关于超平面的几何间隔为：数据集中所有样本点的几何间隔最小值：
$$\gamma=min_{i=1,2,...,m}\gamma_i$$
{{</keepit>}}

### 6.1 支持向量机

模型角度：给定线性可分数据集X，支持向量机模型希望求得数据集X关于超平面的几何间隔 $\gamma$ 达到最大的那那个超平面，然后套上一个sign函数实现分类功能（简单理解，该函数就是值大于0取1，小于0取-1）

{{<keepit>}}
$$y=sign(\omega^Tx+b)=
\begin{cases} 
1, \ \omega^Tx+b>0 \\ 
-1, \ \omega^Tx+b <0 
\end{cases}$$

所起其本质和感知机一样，仍然是在求一个超平面。那么几何间隔最大的超平面一定是前面所说的那个距离正负样本点都最远的超平面。因为：
* 当超平面没有正确划分正负样本时：几何间隔最小的为误分类点，因此 $\gamma < 0$
* 当超平面正确划分超平面时：$\gamma \eqslantgtr 0$，且越靠近中央其值越大。

策略角度：给定线性可分数据集X，设X中集合间隔最小的样本为$(x_min, y_min)$，那么支**持向量机找超平面的过程**可以转化为以下带约束条件的优化问题

首先我们要找到数据集关于超平面的最大几何间隔
$$max \ \ \ \gamma$$
对于每个样本关于超平面的几何间隔中要找到最小的
$$s.t. \ \ \  \gamma_i \eqslantgtr \gamma, \ \ i = 1,2,...,m$$

最小的几何间隔
$$max_{\omega,b} \ \ \frac{y_{min}(\omega^Tx_{min}+b)}{||\omega||}$$
$$s.t. \ \ \ \frac{y_i(\omega^Tx_i+b)}{||\omega||} \eqslantgtr \frac{y_{min}(\omega^Tx_{min}+b)}{||\omega||}, \ \ i=1,2,...,m$$

在约束中将分母$\omega$约掉
$$max_{\omega,b} \ \ \ \frac{y_{min}(\omega^Tx_{min}+b)}{||\omega||}$$
$$s.t. \ \ \ y_i(\omega^Tx_i+b) \eqslantgtr y_{min}(\omega^Tx_{min}+b), \ \ i=1,2,...,m$$

假设该问题的最优解为($\omega^*,b^*$)，那么$(\alpha \omega^*, \alpha b^*), \alpha \in R^+$ 也是最优解，且超平面也不变，因此还需要对$\omega, b$做一定限制才能使得上述优化问题有可能的唯一解。

不防令$y_{min}(\omega^Tx_{min}+b)=1$（这里是固定分子，等于1这个值可以随便取，2、100都行，只要能解出来就行），因为对于特定的(x_{min},y_{min})来说，能使得$y_{min}(\omega^Tx_{min})=1$的$\alpha$有且只有一个（ymin、xmin是固定的且唯一的，如果再将取值固定了，那么w、b就也是固定的了）。因此上述优化问题进一步转化为：

$$max_{\omega,b} \ \ \ \ \frac{1}{||\omega||}$$
$$s.t. \ \ \ \ y_i(\omega^Tx_i+b) \eqslantgtr 1, \ \ i=1,2,...,m$$

为了方便后续计算，将最大化问题转成最小化，其中平方是为了方便计算，二分之一是为了在求导的时候方便约掉，即
$$max_{\omega,b} \ \ \ \ \frac{1}{2} ||\omega||^2$$
$$s.t. \ \ \ \ 1-y_i(\omega^Tx_i+b) \eqslantless 0, \ \ i=1,2,...,m$$
{{</keepit>}}

此优化问题含不等式约束的优化问题，且为凸优化问题，因此可以直接用很多专门求解凸优化问题的方法求解。在这里支持向量机通常采用拉格朗日对偶来求解。

### 6.2 对偶问题

对最大化间隔可以用拉格朗日乘子法得对偶问题 dual problem

#### 6.2.1 基础知识：凸优化问题

对于一般地约束优化问题：
{{<keepit>}}
$$min \ \ \ f(x)$$

$$s.t. \ \ \  
\begin{cases} g_i(x) \eqslantless 0 \ \ \ i=1,2,...,m \\
h_j(x) = 0 \ \ \ j=1,2,...,n
\end{cases}
$$
{{</keepit>}}

若目标函数发f(x)是凸函数，约束集合是凸集，则称为凸优化问题，特别地，g是凸函数，h是线性函数时，约束集合和为凸集，该优化问题为凸优化问题。显然，支持向量机的目标函数$\frac{1}{2}||\omega||^2$是关于w的凸函数，不等式约束$1-y_i(\omega^Tx_i+b)$是关于w的凸函数，因此支持向量机是一个凸优化问题。

拉格朗日对偶不一定用来解决凸优化问题，主要是是对一般地约束优化问题

{{<keepit>}}
设上述优化问题的定义域D是f、g、h三个函数定义域关于x的交集，可行集D^~在属于交集D的基础上，gi(x) <= 0，hj(x) = 0。显然D^~是D的自己，最优质为 $p^*=min{f(\hat{x})}$。由拉格朗日函数的定义可知上述优化问题的拉格朗日函数为
$$L(x,\mu,\lambda) = f(x) + \sum^m_{i=1} \mu_i g_i(x) + \sum^n_{j=1} \lambda_j h_j(x)$$

其中$\mu=(\mu_1,\mu_2,...,\mu_n)^T, \ \lambda = (\lambda_1, \lambda_2,...,\lambda_n)^T$为拉格朗日乘子向量。

定义上述优化问题的拉格朗日对偶函数$F(\mu,\lambda)$（注意其自变量不包含x）为$L(x,\mu,\lambda)$关于x的下确界（inf(e^x)=0，其中0就是e^x函数的下确界，永远也去不到的最小值），也即：

$$F(\mu,\lambda) = {inf}_{x\in D} L(x,\mu,\lambda) = inf_{x\in D} \Bigg( f(x) + \sum^m_{i=1} \mu_i g_i(x) + \sum^n_{j=1} \lambda_j h_j(x) \Bigg)$$

对偶函数F有如下性质：

* 无论是否是凸优化问题，其对偶函数F恒为凹函数
* 当lambda >= 0时，F构成了上述优化问题最优质p^*的下界，也即
$$F(\mu,\lambda) \eqslantless p^*$$
{{</keepit>}}

为什么支持向量机通常都采用拉格朗日对偶求解？

1. 因为无论主问题是什么优化问题，对偶问题一定是凸优化问题，凸优化问题是优化问题中最好解的。原始问题的时间复杂度和特征维数成正比（因为未知量是omega），而对偶问题和数据量成正比（因为未知量是alpha），当特征维数远高于数据量的时候拉格朗日对偶更高效。
2. 对偶函数能很自然引入核函数，进而推广到非线性分类问题（主要原因）

### 6.3 软间隔

上述都是线性可分的问题，而现实中，大部分是线性不可分的问题，因此需要允许支持向量机犯错（软间隔）。

从数学角度来说，软间隔就是允许部分样本（也就是异常数据，尽可能要少）不满足下式中的约束条件：

{{<keepit>}}
$$min_{\omega,b} \ \ \ \frac{1}{2}||\omega||^2$$
$$s.t. \ \ \ y_i(\omega^Tx_i+b) \eqslantgtr 1, \ \ \ i=1,2,...,m$$

因此，可以将必须严格执行的约束条件转化为具有一定灵活性的损失，合格的损失函数要求如下：

* 当满足约束条件时，损失为0
* 当不满足约束条件时，损失不为0
* （可选）当不满足约束条件时，损失与其违反约束条件的成都成正比

只要满足上述要求，才能保证最小化损失的过程中，保证不满足约束条件的样本尽可能的少。

$$min_{\omega,b} \frac{1}{2} ||\omega||^2 + C\sum^m_{i=1} l_{0/1} (y_i(\omega^T x_i +b)-1)$$

其中，l(0/1)是0/1损失函数

$$ l_{0/1}(z) =
\begin{cases}
1, \ \ \ if z < 0 \\
0, \ \ \ if z \eqslantgtr 0
\end{cases}
$$

C>0是一个常数，用来调节损失的权重，显然当$C \rightarrow + \infin$时，会迫使所有样本的损失为0，进而退化为严格执行的约束条件，退化为硬间隔，因此，本世子可以看作支持向量机的一般化形式。

由于l0/1非凸、非连续，数学性质不好，使得上式不易求解，因此常用一些数学性质较好的替代损失函数来代替，软间隔支持向量机通常采用的是hinge（合页）损失来代替：(合页损失函数是一个连续的凸函数)

$$hinge损失：l_{hinge}(z)=max(0,1-z)$$

替换进上式可得：

$$min_{\omega,b} \frac{1}{2} ||\omega||^2 + C \sum^m_{i=1} max(0,1-y_i(\omega^Tx_i+b))$$

引入松弛变量 $\xi$，上述优化问题遍和下述优化问题等价

$$min_{\omega, b, \xi_i} \ \ \ \ \frac{1}{2} ||\omega||^2 + C \sum^m_{i=1} \xi_i$$
$$s.t. 
\begin{cases}
y_i(\omega^Tx_i +b) \eqslantgtr 1- \xi_i \\
\xi_i \eqslantgtr 0, i = 1,2,...,m
\end{cases}
$$

证明：令
$$max(0,1-y_i(\omega^T x_i +b)) = \xi_i$$
显然 $\xi_i \eqslantgtr 0$，当 $1-y_i(\omega^Tx_i+b) > 0 \Rightarrow \xi_i = 1 - y_i(\omega^Tx_i+b)$，当 $1-y_i(\omega^Tx_i+b) \eqslantless 0 \Rightarrow \xi_i = 0$，所以

$$1-y_i(\omega^Tx_i+b) \eqslantless \xi_i \Rightarrow y_i(\omega^Tx_i+b) \eqslantgtr 1-\xi_i$$
{{</keepit>}}

### 支持向量回归

相比于线性回归用一条线来拟合训练样本，支持向量回归（SVR）而采用一个以 $f(x)=\omega^Tx+b$为中心，宽度为$2\epsilon$的**间隔带**，来拟合训练样本。

落在带子上的样本不计算损失（类比线性回归在线性上的点预测误差为0），不在带子上的则偏离带子的距离作为损失（类比线性回归的均方误差），然后以最小化损失的方式破事间隔带从样本最密集的地方（中心地带）穿过，进而达到拟合训练样本的目的。

因此SVR优化问题可以写成

{{<keepit>}}
$$min_{\omega, b} \frac{1}{2} ||\omega||^2 + C \sum^m_{i=1} l_\epsilon (f(x_i) - y_i)$$

其中$l_\epsilon (z)$为$\epsilon$不敏感损失函数（类比均方误差损失）
$$ l_\epsilon (z) = 
\begin{cases}
0, \ \ \ if|z| \eqslantless \epsilon \\
|z|-\epsilon, \ \ \  if|z| > \epsilon
\end{cases}
$$

$\frac{1}{2}||\omega||^2$为L2正则项，此处引入正则项除了起正则化本身的作用外，也是为了和（软间隔）支持向量机的优化目标保持形式上的一致（在这里不用均方误差也是此目的），这样就可以导出对偶函数问题引入核函数，C为调节损失权重的常数。

同软间隔支持向量机，引入松弛$\xi_i$，令
$$l_{\epsilon}(f(x_i)-y_i)=\xi_i$$

显然$\xi_i \eqslantgtr 0$，并且

当 $|f(x_i)-y_i| \eqslantless \epsilon \Rightarrow \xi_i = 0$

当 $|f(x_i)-y_i| > \epsilon \Rightarrow \xi_i = |f(x_i)-y_i|-\epsilon$

所以

$$|f(x_i) - y_i| - \epsilon \eqslantless \xi_i \Rightarrow |f(x_i) - y_i| \eqslantless \epsilon + \xi_i$$

$$-\epsilon-\xi_i \eqslantless f(x_i)-y_i \eqslantless \epsilon+\xi_i$$

那么SVR的优化问题可以改写为

$$min_{\omega,b,\xi_i} \frac{1}{2} ||\omega||^2 + C \sum^m_{i=m} \xi_i$$

$$s.t. \ \ \ -\epsilon-\xi_i \eqslantless f(x_i)-y_i \eqslantless \epsilon + \xi_i \\
\xi_i \eqslantgtr0,i=1,2,...,m$$

如果考虑两边采用不同的松弛程度

$$min_{\omega,b,\xi_i,\hat{\xi_i}} \frac{1}{2} ||\omega||^2 + C \sum^m_{i=m} (\xi_i+\hat{\xi_i})$$

$$s.t. \ \ \ -\epsilon-\hat{\xi_i} \eqslantless f(x_i)-y_i \eqslantless \epsilon + \xi_i \\
\xi_i \eqslantgtr0,\hat{\xi_i} \eqslantgtr0,i=1,2,...,m$$
{{</keepit>}}

