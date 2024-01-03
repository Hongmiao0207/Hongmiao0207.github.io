# Machine Learning Notes 01


## 1. 绪论

### 1.1 基本术语

* data set
* instance / sample
* attribute / feature
* attribute value
* attribute space / sample space
* feature vector
* dimensionality
* learning / training
* training data
* training sample
* training set
* hypothesis: 模型对应了关于数据的某种潜在的规律
* ground-truth: 潜在规律的本身
* learner: 模型，给定数据和参数空间上的实例化
* prediction
* label
* example, ($x_i, y_i$)
* label space
* classification, 分类、离散
* regression, 连续
* binary classification
* positive / negative class
* multi-class classification
* testing
* testing sample
* clustering, 将西瓜做聚类，即将训练集中的西瓜分成若干组
* cluster, 每组称为一个簇
* supervised learning, 分类、回归问题
* unsupervised learning, 聚类
* generalization, 泛化，模型适用于新样本的能力
* distribution, 通常假设样本空间中全体样本服从一个未知**分布**
* independent and identically distributed, 每个样本都是独立地从这个分布上采样获得的

### 1.2 假设空间

科学推理手段：

* induction, 从具体事实归结出一般性规律。
* deduction, 从技术原理推演具体情况。

假设空间：存在一个所有可能的由输入空间到输出空间的映射所构成的集合，则假设空间就是该机喝的一个子集。

### 1.3 归纳偏好

归纳偏好是算法在学习过程中对某种类型假设的偏好，防止算法被假设空间中看似等效的假设过迷惑，而无法产生确定的结果。

## 2. 模型评估和选择

### 2.1 经验误差与过拟合

经验/训练误差，训练集上实际预测输出与样本的真实输出之间的差异。

过拟合，在训练样本表现过好，但是泛化性能下降。一般由于学习能力过于强大导致。

### 2.2 评估方法

#### 2.2.1 留出法 hold-out

将数据集分为两个互斥的集合，一个训练集、一个测试集。两个集合要尽量保持数据分布的一致性。复杂度高。

#### 2.2.2 交叉验证法 cross validation

将数据集划分为k个大小相似的互斥子集，保证一致性。k-1个子集作为训练集，余下的并集作为测试集。

#### 2.2.3 自助法 bootstrapping

适用于数据集小、难以有效划分训练、测试集。改变了初始数据集的分布。在数据量足够时，不如前两个常用。

### 2.3 性能度量 performance measure

衡量模型泛化能力的评价标准，比如回归任务常用的“均方误差”。

#### 2.3.1 错误率和精度

错误率与精度的关系为 $1-E(f;D)=acc(f;D)$

#### 2.3.2 查准率 precision、查全率 recall 与F1

在类似信息检索的应用中，错误率和精度是不够用的。

二分类问题中，根据真实类别和学习器预测类别划分出真正例（true positive）、假正例（false positive）、真反例（true negative）、假反例（false negative），查准率P和查全率R分别定义为：
$$P=\frac{TP}{TP+FP}$$
$$R=\frac{TP}{TP+FN}$$

二者是矛盾的度量，一个高另一个就低。平衡点（Break-Event Point）就是一个综合考虑双方性能度量，取得相对双高的比例。

#### 2.3.3 ROC 与 AUC

ROC (Receiver Operating Characteristic) 受试者工作特征曲线，横坐标是False Positive，纵坐标时 True Positive。其横纵坐标是没有相关性的，所以不能把其当作函数曲线分析，二十四将其看作无数的点。

AUC (Area Under ROC Curve), ROC曲线下的面积。当一个学习器的ROC曲线被另一个学习器的曲线完全包住，可断言后者性能优越。当二者ROC曲线发生交叉时，就需要通过AUC来判据。AUC估算为：
$$AUC=\frac{1}{2}\sum^{m-1}_{i=1}(x_{i+1}-x_i)(y_i+y_{i+1})$$ 

AUC考虑的是样本预测的排序质量，因此它与排序误差有关联。

#### 2.3.4 代价敏感错误率和代价曲线

根据不同类型造成的不同损失，可为错误赋予非均等代价 unequal cost。

在非均等代价下，ROC曲线不能直接反应学习器的期望总体代价，但是代价曲线可以。

### 2.4 偏差与方差

偏差方差分解是解释学习算法泛化性能的重要工具，对算法的期望泛化错误率进行拆解。

