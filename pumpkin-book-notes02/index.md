# Machine Learning Notes 02


## 3 线性模型

### 3.1 基本形式

线性模型形式简单易于建模，一些非线性模型也可在此基础上引入层级结构或高维映射得到。

{{<keepit>}}
$$f(x)= \omega^Tx+b$$
{{</keepit>}}

### 3.2 Linear Regression

线性回归通过给定数据晒图学得一个线性模型达到 $f(x)= \omega^Tx+b$ 无限趋近于 y。

其中$\omega$ 和 b 取值的关键在于衡量f(x)和y之间的差别，因此可以将军方误差最小化来达到目的，即

{{<keepit>}}
$$(\omega^*, b^*) = \underset{(\omega, b)}{\arg\min} \sum_{i=1}^{m} (f(x_i) - y_i)^2$$

$$=arg \, min_(\omega,b) \sum^m_{i=1} (y_i-\omega x_i-b)^2$$
{{</keepit>}}

均方误差的几何意义是欧氏距离，其求解方式是最小二乘法，找到一条直线使所有样本到该直线上的欧氏距离之和最小。

线性回归模型的最小二乘“参数估计”是求解$\omega$和b是均方误差 $E_{(\omega,b)}=\sum^m_{i=1}(y_i-\omega x_i-b)^2$ 最小化的过程。E是关于$\omega$和b的凸函数，当导数为0时，得到最优解。

注：凸函数，对区间[a, b]上定义的函数f，若它对区间中任意两点x1，x2均有 $f(\frac{x_1+x_2}{2} <= \frac{f(x_1)+f(x_2)}{2})$，则称f为区间[a,b]上的凸函数。

* U醒曲线的函数如f(x)=x^2，通常为凸函数。
* 实数集上的函数，可通过求二阶导来判别：二阶导在区间上为非负就是凸函数；若其恒大于0，则称为严格凸函数。

#### 3.2.1  一元线性回归

* 3.5 对 $\omega$求导:

{{<keepit>}}
$$\frac{\partial E_{(\omega, b)}}{\partial \omega}=2 \Bigg(\omega \sum^m_{i=1}x^2_i - \sum^m_{i=1}(y_i - b) x_ix \Bigg)$$
{{</keepit>}}
  
* 推导：

已知 $E_{(\omega,b)}=\sum^m_{i=1}(y_i-\omega x_i-b)^2$，所以

{{<keepit>}}
$$\frac{\partial E_{(\omega, b)}}{\partial \omega}= \frac{\partial}{\partial \omega} \Bigg[ \sum^m_{i=1} (y_i - \omega x_i - b)^2 \Bigg]$$

$$= \sum^m_{i=1} \frac{\partial}{\partial \omega} \Bigg[ (y_i - \omega x_i - b)^2 \Bigg] $$

$$= \sum^m_{i=1} [2 (y_i - \omega x_i - b) (-x_i)]$$

$$= \sum^m_{i=1} [2 (\omega x_i^2 - y_ix_i +bx_i)]$$

$$= 2 \Bigg( \omega \sum^m_{i=1} x_i^2 - \sum^m_{i=1} y_ix_i + b \sum^m_{i=1} x_i \Bigg)$$

$$= 2 \Bigg( \omega \sum^m_{i=1} x_i^2 - \sum^m_{i=1} (y_i - b)x_i \Bigg)$$
{{</keepit>}}

* 3.6 对b求导
{{<keepit>}}
$$\frac{\partial E_{(\omega, b)}}{\partial b} = 2 \Bigg( mb - \sum^m_{i=1} (y_i - \omega x_i) \Bigg)$$
{{</keepit>}}

* 推导：

已知 $E_{(\omega,b)}=\sum^m_{i=1}(y_i-\omega x_i-b)^2$，所以

{{<keepit>}}
$$\frac{\partial E_{(\omega, bn)}}{\partial b} = \frac{\partial}{\partial b} \Bigg[ \sum^m_{i=1} (y_i - \omega x_i - b)^2 \Bigg]$$

$$= \sum^m_{i=1} \frac{\partial}{\partial b} \Bigg[ (y_i - \omega x_i - b)^2 \Bigg]$$

$$= \sum^m_{i=1} [2 (y_i - \omega x_i - b)(-1)] $$

$$= \sum^m_{i=1} [2(b - y_i + \omega x_i)]$$

$$= 2 \Bigg[ \sum^m_{i=1} b - \sum^m_{i=1} y_i + \sum^m_{i=1} \omega x_i \Bigg]$$

$$= 2 \Bigg( mb - \sum^m_{i=1}(y_i - \omega x_i) \Bigg)$$
{{</keepit>}}

* 当3.5、3.6为0得到 $\omega$ 和 b 最优解的闭式解（closed-form）

3.7

{{<keepit>}}
$$\omega = \frac{\sum^m_{i=1} y_i (x_i - \bar{x})}{\sum^m_{i=1} x^2_i - \frac{1}{m} (\sum^m_{i=1} x_i)^2}$$
{{</keepit>}}

3.8

{{<keepit>}}
$$b = \frac{1}{m} \sum^m_{i=1} (y_i - \omega x_i)$$
{{</keepit>}}

其中 $\bar{x} = \frac{1}{m} \sum^m_{i=1} x_i$ 为 x 的均值。

3.7推导，令3.5等于0

{{<keepit>}}
$$0 = \omega \sum^m_{i=1} x^2_i - \sum^m_{i=1} (y_i - b) x_i$$

$$\omega \sum^m_{i=1} x^2_i = \sum^m_{i=1} y_i x_i - \sum^m_{i=1}  b x_i$$
{{</keepit>}}

由于令3.6等于0可得 $b = \frac{1}{m} \sum^m_{i=1} (y_i - \omega x_i)$，又因为 $\frac{1}{m} \sum^m_{i=1} y_i = \bar{y}$ ，$\frac{1}{m} \sum^m_{i=1} x_i = \bar{x}$, 则 $b = \bar{y} - \omega \bar{x}$，代入上式可得：

{{<keepit>}}
$$\omega \sum^m_{i=1} x^2_i = \sum^m_{i=1} y_i x_i - \sum^m_{i=1} (\bar{y} - \omega \bar{x}) x_i$$

$$\omega \sum^m_{i=1} x^2_i = \sum^m_{i=1} y_i x_i - \bar{y} \sum^m_{i=1} x_i + \omega \bar{x} \sum^m_{i=1} x_i$$

$$\omega (\sum^m_{i=1} x^2_i - \bar{x} \sum^m_{i=1} x_i) = \sum^m_{i=1} y_i x_i - \bar{y} \sum^m_{i=1} x_i$$

$$\omega = \frac{\sum^m_{i=1} y_i x_i - \bar{y} \sum^m_{i=1} x_i}{\sum^m_{i=1} x^2_i - \bar{x} \sum^m_{i=1} x_i}$$
{{</keepit>}}

由于

{{<keepit>}}
$$\bar{y} \sum^m_{i=1} x_i = \frac{1}{m} \sum^m_{i=1}y_i \sum^m_{i=1} x_i = \bar{x} \sum^m_{i=1} y_i$$

$$\bar{x} \sum^m_{i=1} x_i = \frac{1}{m} x_i \sum^m_{i=1} x_1 = \frac{1}{m}(\sum^m_{i=1}x_i)^2$$
{{</keepit>}}

代入上式即可得到公式3.7

{{<keepit>}}
$$\omega = \frac{\sum^m_{i=1} y_i x_i - \bar{y} \sum^m_{i=1} x_i}{\sum^m_{i=1} x^2_i - \bar{x} \sum^m_{i=1} x_i}$$

$$= \frac{\sum^m_{i=1} y_i x_i - \bar{x} \sum^m_{i=1} y_i}{\sum^m_{i=1} x^2_i - \frac{1}{m}(\sum^m_{i=1}x_i)^2}$$
{{</keepit>}}

3.7

{{<keepit>}}
$$= \frac{\sum^m_{i=1} y_i (x_i-\bar{x})}{\sum^m_{i=1} x^2_i - \frac{1}{m}(\sum^m_{i=1}x_i)^2}$$
{{</keepit>}}

上述的求和运算只能通过python的循环来实现，如果将上式向量化，转换成矩阵运算，那就可以通过 Numpy来实现。向量化：

将 $\frac{1}{m}(\sum^m_{i=1} x_i)^2 = \bar{x} \sum^m_{i=1} x_i$ 代入分母可得：

{{<keepit>}}
$$\omega = \frac{\sum^m_{i=1} y_i (x_i - \bar{x})}{\sum^m_{i=1} x^2_i - \bar{x} \sum^m_{i=1} x_i}$$

$$=\frac{\sum^m_{i=1} (y_ix_i - y_i\bar{x})}{\sum^m_{i=1} (x^2_i - \bar{x} x_i)}$$
{{</keepit>}}

又因为

{{<keepit>}}
$$\bar{y} \sum^m_{i=1} x_i = \bar{x} \sum^m_{i=1}y_i = \sum^m_{i=1} \bar{y}x_i = \sum^m_{i=1} \bar{x}y_i = m\bar{x} \bar{y} = \sum^m_{i=1} \bar{x}\bar{y}$$

$$\sum^m_{i=1} x_i \bar{x} = \bar{x} \sum^m_{i=1} x_i = \bar{x}m\frac{1}{m} \sum^m_{i=1} x_i = m \bar{x}^2 = \sum^m_{i=1} \bar{x}^2$$
{{</keepit>}}

则上式可化为：

{{<keepit>}}
$$\omega = \frac{\sum^m_{i=1} (y_ix_i - y_i \bar{x} - x_i \bar{y} + \bar{x} \bar{y})}{\sum^m_{i=1}(x^2_i - x_i\bar{x} - x_i\bar{x} + \bar{x}^2)}$$

$$= \frac{\sum^m_{i=1} (x_i - \bar{x}) (y_i - \bar{y})}{\sum^m_{i=1} (x_i - \bar{x})^2}$$
{{</keepit>}}

若令 $x = (x_1,x_2,...,x_m)^T$，$x_d=(x_1-\bar{x}, x_2-\bar{x},...,x_m-\bar{x})^T$ 为去均值后的x。 $y = (y_1,y_2,...,y_m)^T$，$y_d=(y_1-\bar{y}, y_2-\bar{y},...,y_m-\bar{y})^T$ 为去均值后的y。其中 x、xd、y、yd均为m行1列的列向量，代入上式可得：

$$\omega = \frac{x^T_d y_d}{x^T_d x_d}$$

#### 3.2.2 多元线性回归 multivariate linear regression

样本由d个属性描述，而不是单个属性描述。多个特征，x和权重就变成了向量。

$$f(x_i) = \omega^Tx_i+b$$

1

{{<keepit>}}
$$f(x_1)=(\omega_1 \, \omega_2 \, \dots \, \omega_d)
\left(
\begin{matrix}
x_{i1} \\
x_{i2} \\
\vdots \\
x_{id}
\end{matrix}
\right)
+b
$$
{{</keepit>}}

2

{{<keepit>}}
$$ f(x_i) = \omega_1 x_{i1} + \omega_2 x_{i2} + ... + \omega_d x_{id} + b$$
{{</keepit>}}

3

{{<keepit>}}
$$f(x_i) = \omega_1 x_{i1} + \omega_2 x_{i2} + ... + \omega_d x_{id} + \omega_{d+1}$$
{{</keepit>}}

解析：将 b 拆成 $\omega_{d+1} * 1$，此时前面是omega，后面的1是x，因此，在1式中，omega_d之后加一个新元素omega_d+1，因此 x_id后面也会多出一个新元素，就是1。所以可知，w_d+1 * 1 = b，所以 2、3式子是等价的。

这样做的好处是可以将式子全部向量化。

{{<keepit>}}
$$
f(x_1)=(\omega_1 \, \omega_2 \, \dots \, \omega_d \, \omega_{d+1})
\left(
\begin{matrix}
x_{i1} \\
x_{i2} \\
\vdots \\
x_{id} \\
1
\end{matrix}
\right)
$$
{{</keepit>}}

$$f(\hat{x}_i) = \hat{\omega}^T \hat{x}_i$$

由最小二乘法可得：

{{<keepit>}}
$$
E_{\hat{\omega}}
= \sum^m_{i=1} (y_i - f(\hat{x}_i))^2
= \sum^m_{i=1}(y_i - \hat{\omega}^T \hat{x}_i)^2
$$
{{</keepit>}}

再将求和符号向量化（向量化的目的是便于用过numpy运算）

1. 将求和拆开

{{<keepit>}}
$$
E_{\hat{\omega}}
= \sum^m_{i=1}(y_i - \hat{\omega}^T \hat{x}_i)^2
= (y_1 - \hat{\omega}^T \hat{x}_1)^2
+ (y_2 - \hat{\omega}^T \hat{x}_2)^2
+ \dots
+ (y_m - \hat{\omega}^T \hat{x}_m)^2
$$
{{</keepit>}}

解析，如何将求和拆成向量形式

{{<keepit>}}
$$
a^2+b^2
= [a \ b]
\left[
\begin{matrix}
a \\
b
\end{matrix}
\right]
$$

$$
E_{\hat{\omega}}
= (y_1 - \hat{\omega}^T \hat{x}_1 \ \ \ y_2 - \hat{\omega}^T \hat{x}_2 \ \ \ ... \ \ \ y_m - \hat{\omega}^T \hat{x}_m)
\left(
\begin{matrix}
y_1 - \hat{\omega}^T \hat{x}_1 \\
y_2 - \hat{\omega}^T \hat{x}_2 \\
\vdots \\
y_m - \hat{\omega}^T \hat{x}_m
\end{matrix}
\right)
$$
{{</keepit>}}

后面的式子化简：

1. 先将拆成两个列向量相减，后面的根据 $a^Tb=b^Ta$ 原则进行转换，因为它最终为一个数，所以相等，可以转换。

{{<keepit>}}
$$
\left(
\begin{matrix}
y_1 - \hat{\omega}^T \hat{x}_1 \\
y_2 - \hat{\omega}^T \hat{x}_2 \\
\vdots \\
y_m - \hat{\omega}^T \hat{x}_m
\end{matrix}
\right)
=
\left(
\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{matrix}
\right)
-
\left(
\begin{matrix}
\hat{\omega}^T \hat{x}_1 \\
\hat{\omega}^T \hat{x}_2 \\
\vdots \\
\hat{\omega}^T \hat{x}_m
\end{matrix}
\right)
=
\left(
\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{matrix}
\right)
-
\left(
\begin{matrix}
\hat{x}^T_1 \hat{\omega} \\
\hat{x}^T_2 \hat{\omega} \\
\vdots \\
\hat{x}^T_m \hat{\omega}
\end{matrix}
\right)
$$

$$y = \left(\begin{matrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{matrix} \right)$$
{{</keepit>}}

1. 每个元素中都有 $\hat{\omega}$，所以提出来。将 $\hat{x}^T_i$ 拆成 $x^T_i \ \ 1$ 的形式，最后一个蒜素恒置为1。西瓜书上定义了一个大X，就是这样的形式。
{{<keepit>}}
$$
\left(
\begin{matrix}
\hat{x}^T_1 \hat{\omega} \\
\hat{x}^T_2 \hat{\omega} \\
\vdots \\
\hat{x}^T_m \hat{\omega}
\end{matrix}
\right)
=
\left(
\begin{matrix}
\hat{x}^T_1 \\
\hat{x}^T_2 \\
\vdots \\
\hat{x}^T_m \\
\end{matrix}
\right)

\hat{\omega}
=
\left(
\begin{matrix}
x^T_1 \ \ 1 \\
x^T_2 \ \ 1 \\
\vdots \ \ \ \ \vdots \\
x^T_m \ \ 1
\end{matrix}
\right)

\hat{\omega}
=
X \hat{\omega}
$$
{{</keepit>}}

因此

{{<keepit>}}
$$
\left(
\begin{matrix}
y_1 - \hat{\omega}^T \hat{x}_1 \\
y_2 - \hat{\omega}^T \hat{x}_2 \\
\vdots \\
y_m - \hat{\omega}^T \hat{x}_m
\end{matrix}
\right)
= y - X \hat{\omega}
$$
{{</keepit>}}

下面的式子中，第二项就是上面的，第一项就是第二项的转置

{{<keepit>}}
$$
E_{\hat{\omega}}
=
\left(
\begin{matrix}
y_1 - \hat{\omega}^T \hat{x}_1 \ \ \
y_2 - \hat{\omega}^T \hat{x}_2 \ \ \
\dots \ \ \
y_m - \hat{\omega}^T \hat{x}_m
\end{matrix}
\right)

\left(
\begin{matrix}
y_1 - \hat{\omega}^T \hat{x}_1 \\
y_2 - \hat{\omega}^T \hat{x}_2 \\
\vdots \\
y_m - \hat{\omega}^T \hat{x}_m  
\end{matrix}
\right)
$$
{{</keepit>}}

完成 损失函数 向量化

{{<keepit>}}
$$
E{\hat{\omega}} = (y-X \hat{\omega})^T (y-X \hat{\omega})
$$
{{</keepit>}}

求解$\hat{\omega}$
$$\hat{\omega}^* = arg_{\hat{\omega}}min (y - X \hat{\omega})^T (y-X\hat{\omega})$$

求解$\omega$仍然是一个多元函数求最值问题，也是凸函数求最值问题。

思路：

1. 证明 $E_{\hat{\omega}} = (y-X \hat{\omega})^T (y-X \hat{\omega})$ 是关于$\hat{\omega}$ 的凸函数。（证明凸函数就是Ew是w的二级偏导数。因为w是向量（多元），也就是求他的海森矩阵。证明该矩阵是一个半正定矩阵即可）
2. 用凸函数求最值的思路求 $\hat{\omega}$。（令它的梯度为0，也就是关于w的一阶导数为0， $\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}} = 0$）

求解：

1. 求 $E_{\hat{\omega}}$ 的Hessian矩阵 $\nabla^2 E_{\hat{\omega}}$（这个倒三角符号就是梯度、向量微分），然后判断其正定性：

{{<keepit>}}
$$
\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}}
=
\frac{\partial}{\partial \hat{\omega}}
[
    (y-X \hat{\omega})^T
    (y-X \hat{\omega})
]
$$

$$
= \frac{\partial}{\partial \hat{\omega}}
[
    (y^T - \hat{\omega}^T X^T)
    (y-X \hat{\omega})
]
$$

$$
= \frac{\partial}{\partial \hat{\omega}}
[
    y^T y - y^T X \hat{\omega} - \hat{\omega}^T X^T y + \hat{\omega} X^T X \hat{\omega}
]
$$

其中，$y^Ty$ 不含有 $\omega$，可以扔掉。

$$
= \frac{\partial}{\partial \hat{\omega}}
[
    - y^T X \hat{\omega} - \hat{\omega}^T X^T y + \hat{\omega} X^T X \hat{\omega}
]
$$

$$
= -\frac{\partial y^T X \hat{\omega}}{\partial \hat{\omega}}
- \frac{\partial \hat{\omega}^T X^T y}{\partial \hat{\omega}}
+\frac{\partial \hat{\omega} X^T X \hat{\omega}}{\partial \hat{\omega}}
$$
{{</keepit>}}

已知 $\hat{\omega}$ 是d+1元的标量函数，也就是说它的未知数是d+1维的向量。但是三个分子上最后函数式子的结果是个标量，因为y转置是个行向量、X是个矩阵，$\hat{\omega}$是个列向量，整体结果是个标量。所以它是**标量关于向量求导** （矩阵微分的内容）。求导方式就是梯度

如何求解 $\hat{\omega}$

* 【标量-向量】的矩阵微分公式：设 $x \in R^{n*1}, f: R^n \rightarrow R$ 为关于x的实值标量函数，则

{{<keepit>}}
$$
\frac{\partial f(x)}{\partial x}
=
\left[
\begin{matrix}
\frac{\partial f(x)}{\partial x_1} \\
\frac{\partial f(x)}{\partial x_2} \\
\vdots \\
\frac{\partial f(x)}{\partial x_n}
\end{matrix}
\right]
$$

$$
\frac{\partial f(x)}{\partial x^T}
=
\left(
\begin{matrix}
\frac{\partial f(x)}{\partial x_1} \ 
\frac{\partial f(x)}{\partial x_2} \
\dots \
\frac{\partial f(x)}{\partial x_n}
\end{matrix}
\right)
$$
{{</keepit>}}

* 就是对x分量（x1...xn）的偏导数求出来组成一个列向量。组成列向量为分母布局（默认），组成行向量为分子布局，仅差一个转置。

求解 $\hat{\omega}$

{{<keepit>}}
$$
\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}}
=
-\frac{\partial y^T X \hat{\omega}}{\partial \hat{\omega}}
- \frac{\partial \hat{\omega}^T X^T y}{\partial \hat{\omega}}
+\frac{\partial \hat{\omega} X^T X \hat{\omega}}{\partial \hat{\omega}}
$$
{{</keepit>}}

根据矩阵微分公式 $\frac{\partial x^T a}{\partial x} = \frac{\partial a^T x}{\partial x} = a$，$\frac{\partial x^T Ax}{\partial x} = (A + A^T)x$ 可得：

{{<keepit>}}
$$
\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}}
= -X^Ty - X^Ty + (X^TX + X^TX) \hat{\omega}
$$

$$= 2X^T(X \hat{\omega - y})$$
{{</keepit>}}

其中，$-\frac{\partial y^T X \hat{\omega}}{\partial \hat{\omega}}$ 这一项是根据 $\frac{\partial a^T x}{\partial x}$ 公式求解的，公式中的x就是式子中的 $\omega$，公式中的是 $a^T$ 就是 $y^T X$，那么 a 就是 $y^T X$的转置 = $X^Ty$

在一阶导数上再求一阶导：

{{<keepit>}}
$$\nabla^2 E_{\hat{\omega}} = \frac{\partial}{\partial \hat{\omega}} \Bigg( \frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}} \Bigg)$$

$$ = \frac{\partial}{\partial \hat{\omega}} [2X^T (X \hat{\omega} - y)] $$

$$= \frac{\partial}{\partial \hat{\omega}} (2X^TX \hat{\omega} - 2X^Ty)$$
{{</keepit>}}

根据矩阵微分公式 $\frac{\partial Ax}{x} = A^T$ 可得：

$$\nabla^2 E_{\hat{\omega}} = 2X^TX$$

西瓜书上假定 $X^TX$ 为**正定矩阵**（本身不一定为），因此Ew是关于w的**凸函数**得证。

令其等于0：

$$\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}} = 2X^T(X \hat{\omega} - y) = 0$$

$$2X^TX \hat{\omega} - 2X^Ty = 0$$

$$2X^TX \hat{\omega} = 2X^Ty$$

3.11
$$\hat{\omega} = (X^TX)^{-1} X^Ty$$

### 3.3 对数几率回归

[扩展阅读](https://sm1les.com/2019/01/17/logistic-regression-and-maximum-entropy/)

对数几率回归就是逻辑回归，本质是分类算法。回归算法主要预测一些具体的数值。

分类算法，比如二分类，预测一个结果的概率。因此它的值是在0-1中，而线性回归值域是在实数域，就不能直接拿来用。因此在线性模型的基础上套用个映射函数来实现分类的目的。

对数几率回归就是套用了一个 $\frac{1}{1+e^{-z}}$这样的函数。它是一个在0-1之间的s型的曲线，又叫做 Sigmoid函数。其中，z就是线性回归的f(x)。

能够将线性函数映射到0-1区间的函数有很多，为什么只用Sigmod函数？原因很多，最有说服力的是从最大熵的角度来解释。

极大似然估计

第一步：确定概率质量函数（概率密度函数）

已知离散型随机变量 $y \in \lbrace 0,1 \rbrace$ 取值为1和0的概率分别建模为

{{<keepit>}}
$$p(y=1|x) = \frac{1}{1+e^{\omega^Tx+b}} = \frac{e^{\omega^T x +b}}{1+e^{\omega^Tx+b}}$$

$$p(y=0|x) = 1 - p(y=1|x) = \frac{1}{1+e^{\omega^Tx+b}}$$
{{</keepit>}}

为了便于讨论，令 $\beta = (\omega; b) \, , \, \hat{x} = (x;1)$，则上式可简写为：

{{<keepit>}}
$$p(y=1|\hat{x}; \beta) = \frac{e^{\beta^T \hat{x}}}{1+e^{\beta^T \hat{x}}} = p_1(\hat{x}; \beta)$$

$$p(y=0|\hat{x}; \beta) = \frac{1}{1+e^{\beta^T \hat{x}}} = p_0(\hat{x}; \beta)$$
{{</keepit>}}

由以上概率取值可推得随机变量 $y \in \lbrace 0,1 \rbrace$ 的概率质量函数为

3.26
$$p(y|\hat{x};\beta) = y p_1(\hat{x}; \beta) + (1-y) p_0(\hat{x}; \beta)$$

或者为

$$p(y|\hat{x};\beta) = [p_1(\hat{x}; \beta)]^y [p_0(\hat{x};\beta)]^{1-y}$$

理解，当y=0，第一个式子前项为0，直接求后项。第二个式子前项为1，也是直接求后项即可，相反也是。

第二步：写出似然函数

$$L(\beta) = \prod^m_{i=1} p (y_i|\hat{x}_i;\beta)$$

对数似然函数为

$$l(\beta) = ln L(\beta) = \sum^m_{i=1} ln p(y_i | \hat{x}_i ; \beta)$$

$$l(\beta) = \sum^m_{i=1}ln(y_ip_1(\hat{x}_i;\beta) + (1 - y_i) p_0 (\hat{x}_i) ; \beta)$$

将 $p_1(\hat{x}_i ; \beta) = \frac{e^{\beta^T \hat{x}_i}}{1+e^{\beta^T \hat{x}_i}}$，$p_0(\hat{x}_i;\beta) = \frac{1}{1 + e^{\beta^T \hat{x}_i}}$ 代入上式可得

$$l(\beta) = \sum^m_{i=1} ln \Bigg( \frac{y_1 e^{\beta^T \hat{x}_1}}{1+e^{\beta^T \hat{x}_i}} + \frac{1-y_i}{1+e^{\beta^T \hat{x}_i}} \Bigg)$$

$$= \sum^m_{i=1} ln \Bigg( \frac{y_i e^{\beta^T \hat{x}_i} + 1 - y_i}{1+e^{\beta^T \hat{x}_i}} \Bigg)$$

根据ln函数规则，$ln \frac{a}{b} = lna - lnb$

$$= \sum^m_{i=1} (ln(y_i e^{\beta^T \hat{x}_1} + 1 - y_i) - ln (1 + e^{\beta^T \hat{x}_i}))$$

由于$y_i \in \lbrace 0,1 \rbrace$，则

$$
l(\beta) = 
\begin{cases}
\sum^m_{i=1}(-ln(1+e^{\beta^T \hat{x}_i}))
, \ \ \ \ \ y_i = 0
\\
\sum^m_{i=1}(\beta^T \hat{x}_i - ln(1 + e^{\beta^T \hat{x}_i}))
, \ \ \ \ \ y_i = 1
\end{cases}
$$

两式综合可得

$$l(\beta) = \sum^m_{i=1} \bigg( y_i \beta^T \hat{x}_i - ln(1+e^{\beta^T \hat{x}_i}) \bigg)$$

为什么可以写成这样？是因为当y=0时，$y_i \beta^T \hat{x}_i = ln(y_i e^{\beta^T \hat{x}_1} + 1 - y_i) = 0$。y=1时，$y_i \beta^T \hat{x}_i = ln(y_i e^{\beta^T \hat{x}_1} + 1 - y_i) = \beta^T \hat{x}_i$。

损失函数通常是以最小化为优化目标，因此可以将最大化 $l(\beta)$ 等价转换为最小化 $l(\beta)$的相反数 $-l(\beta)$，即得到公式3.27。

3.27
$$l(\beta) = \sum^m_{i=1} \bigg( - y_i \beta^T \hat{x}_i - ln(1+e^{\beta^T \hat{x}_i}) \bigg)$$

#### 拓展：信息论

以概率论、随机过程为基本研究工具，研究广义通信系统的整个过程。常见应用：无损数据压缩（zip）、有损数据压缩（MP3、JPEG）

**自信息**：随机变量x，它有一个概率质量函数p(x)，它的自信息就是负的log底数为b的函数。

$$I(X) = -log_b p(x)$$

当b=2时单位为bit，当b=e时单位为nat。（为e的时候就是ln）

**信息熵**（自信息的期望）：度量随机变量X的不确定性，信息熵越大越不确定。

$H(X) = E[I(X)] = - \sum_x p(x) log_b p(x)$ （以离散型为例）

计算信息熵时约定：若p(x)=0，则p(x)log_b p(x)=0 （具体在决策树讲解）

**相对熵** （KL散度）：度量两个分布的差异，其典型使用场景是用来度量理想分布 p(x) 和模拟分布 q(x) 之间的差异。

$$D_{KL}(p||q) = \sum_x p(x) log_b (\frac{p(x)}{q(x)})$$
$$= \sum_x p(x) (log_b p(x) - log_bq (x))$$
$$= \sum_x p(x) log_b p(x) - \sum_x p(x) log_b q(x)$$

其中，$\sum_x p(x) log_b p(x)$ 没加负号的理想分布的信息熵，$- \sum_x p(x) log_b q(x)$ 称为交叉熵，$\sum_x$ 就是遍历x所有可能的取值，。

**如何理交叉熵和信息熵？**

* 从机器学习三要素中"策略"的角度来说，与理想分布**最接近的模拟分布**即为最优分布，因此可以通过**最小化相对熵**这个策略来求出最优分布，原因是当相对熵达到最小时，q(x)最接近p(x)。由于理想分布p(x)是未知但固定分布（频率学派角度），所以 $\sum_x p(x) log_b p(x)$为常量，那么最小化相对熵就等价于最小化交叉熵 $- \sum_x p(x) log_b q(x)$。

以对数几率回归为例，对单个样本yi来说，理想分布

$$
p(y_i) =
\begin{cases}
p(1)=1, p(0) = 0, \ \ \ \ y_i=1 \\
p(1)=0, p(0)=1, \ \ \ \ y_i=0
\end{cases}
$$

它现在的模拟分布是

$$
q(y_1) = 
\begin{cases}
\frac{e^{\beta^T \hat{x}}}{1+e^{\beta^T \hat{x}}} = p_1(\hat{x}; \beta), \ \ \ \ y_i=1 \\
\frac{1}{1+e^{\beta^T \hat{x}}} = p_0(\hat{x}; \beta), \ \ \ \ y_i =0
\end{cases}
$$

那么单个样本yi的交叉熵为

$$-\sum_{y_i} p(y_i) log_b q(y_i)$$ 

$$-p(1) log_b p_1 (\hat{x}; \beta) - p(0) log_b p_0 (\hat{x}; \beta)$$

$$-y_i log_bp_1(\hat{x}; \beta) - (1 - y_i) log_b p_0 (\hat{x}; \beta)$$

令 b=e

$$-y_i ln p_1 (\hat{x}; \beta) - (1- y_i) ln p_0 (\hat{x}; \beta)$$

全体训练样本的交叉熵

$$\sum^m_{i=1} [-y_i ln p_1 (\hat{x}; \beta) - (1-y_i) ln p_0(\hat{x}; \beta)]$$

$$\sum^m_{i=1} \lbrace -y_i [ln p_1 (\hat{x}_i; \beta) - ln p_0 (\hat{x}_i ; \beta)] - ln(p_0(\hat{x}_i; \beta)) \rbrace$$

$$\sum^m_{i=1} \Bigg[ -y_iln \bigg( \frac{p_1(\hat{x}_i;\beta)}{p_0(\hat{x}_i;\beta)}  - ln(p_0(\hat{x}_i;\beta))\bigg) \Bigg]$$

$$\sum^m_{i=1} \Bigg[ -y_iln \bigg( \frac{\frac{e^{\beta^T \hat{x}}}{1+e^{\beta^T} \hat{x}}}{\frac{1}{1+e^{\beta^T} \hat{x}}}  - ln(\frac{1}{1+e^{\beta^T} \hat{x}})\bigg) \Bigg]$$

$$\sum^m_{i=1} \Bigg[ -y_i ln(e^{\beta^T} \hat{x}_i) - ln(\frac{1}{1+e^{\beta^T} \hat{x}}) \Bigg]$$

又得到了公式3.27

$$\sum^m_{i=1} (-y_i \beta^T \hat{x}_i) + ln(1+e^{\beta^T} \hat{x}_i)$$

对数几率回归算法的机器学习三要素

1. 模型：线性模型，输出值在[0,1]，近似阶跃的单调可微函数
2. 策略：极大似然，信息论
3. 算法：梯度下降，牛顿法

### 3.4 二分类线性判别分析

异类样本尽可能远，同类样本方差尽可能小。

在西瓜书定义中，假定了一个数据集 D={(xi, yi)}，其中，i是从1取到m，yi属于0到1。要注意的是Xi，大X的i和数据集中的i不是一个东西。X中i是0，1，X1表示所有y=1的[(xi,yi)]的集合。X0就是y=0的集合。

$||a||^2_2$，这种形式叫做二范数，求的是向量的模长。比如，一个向量 $a=(a_1,a_2)^T$，那它的二范数就是 $||a||_2 = \sqrt{{a_1}^2 + {a_2}^2}$，上面再加个2就是平方 $||a||_2^2 = {a_1}^2 + {a_2}^2$。

