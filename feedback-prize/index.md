# Feedback Prize - Evaluating Student Writing | Kaggle


## 1 比赛介绍

在这次竞赛中，将识别学生写作中的元素。更具体地说，将自动分割文本，并对6-12年级学生所写的文章中的论证和修辞元素进行分类。帮助学生提高写作水平的一个方法是通过自动反馈工具，评估学生的写作并提供个性化的反馈。

- 竞赛类型：本次竞赛属于深度学习/自然语言处理，所以推荐使用的模型或者库：Roberta/Deberta/Longformer
- 赛题数据：官方提供的训练集大约有15000篇文章，测试集大约有10000篇文章。然后将分割的每个元素分类为以下内容之一：引子/立场/主张/反诉/反驳/证据/结论性声明，值得注意的是，文章的某些部分将是未加注释的（即它们不适合上述的分类）。
- 评估标准：标签和预测的单词之间的重合度。通过计算每个类别的TP/FP/FN，然后取所有类别的 macro F1 score 分数得出。详见：[Feedback Prize - Evaluating Student Writing | Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021/overview/evaluation)。
- 推荐阅读 Kaggle 内的一篇 EDA（探索数据分析）来获取一些预备知识：[NLP on Student Writing: EDA | Kaggle](https://www.kaggle.com/code/erikbruin/nlp-on-student-writing-eda)。

### 1.1 数据说明

官方提供的训练集大约有15000篇文章，测试集大约有10000篇文章。然后将分割的每个元素分类为以下内容之一：引子/立场/主张/反诉/反驳/证据/结论性声明，值得注意的是，文章的某些部分将是未加注释的（即它们不适合上述的分类）。

官方数据页面：[Feedback Prize - Evaluating Student Writing | Kaggle](https://www.kaggle.com/competitions/feedback-prize-2021/data)。

将分割的每个元素分类为以下内容之一：

- **引子**–以统计数字、引文、描述或其他一些手段开始的介绍，以吸引读者的注意力并指向论题。
- **立场**–对主要问题的看法或结论。
- **主张**–支持该立场的主张。
- **反诉**–反驳另一个诉求的诉求，或提出与立场相反的理由。
- **反驳**–驳斥反诉的主张。
- **证据**–支持主张、反主张或反驳的观点或例子。
- **结论性声明**–重申主张的结论性声明。

值得注意的是，文章的某些部分将是未加注释的（即它们不适合上述的分类）。

train.csv - 一个包含训练集中所有论文注释版本的.csv文件：

- **id** - 作文的ID。
- **discourse_id** - 话语元素的ID。
- **discourse_start** - 话语元素在文章中开始的字符位置。
- **discourse_end** - 话语元素在文章中结束的位置。
- **discourse_text** - 话语元素的文本。
- **discourse_type** - 话语元素的分类。
- **discourse_type_num** - 话语元素的列举式分类标签（带序号）。
- **predictionstring** - 训练样本的词索引，为预测所需。

### 1.2 评价标准

评估依据是：**标签** 和 **预测** 的单词之间的重合度。

对于每个样本，所有的标签和对某一特定类别的预测都要进行比较。

- 如果标签和预测之间的重合度>=0.5，而预测和标签之间的重合度>=0.5，则预测是一个匹配，被认为是一个真阳性。
- 如果存在多个匹配，则取重叠度最高的一对匹配。
- 任何没有匹配的标签是假阴性，任何没有匹配的预测是假阳性。

#### 1.2.1 举例

标签：

![label01](/posts/kaggle/feedback-prize/label01.png)

预测：

![prediction01](/posts/kaggle/feedback-prize/prediction01.png)

- 第一个预测与任何一个标签都没有>=0.5的重叠，是一个假阳性。
- 第二个预测将与第二个标签完全重叠，是一个真阳性。
- 第三个标签将是不匹配的，是一个假阴性。

最后的分数是通过计算每个类别的TP/FP/FN，然后取所有类别的 [macro F1 score](https://en.wikipedia.org/wiki/F-score) 分数得出的。

词的索引是通过使用Python的.split()函数计算的，并在得到的列表中获取索引。

两个重叠部分的计算方法是：取标签/预测对中的每个指数列表的set()，并计算两个集合的交点除以每个集合的长度。

**F1-score**：

$$ F_1 = \frac{2}{recall^{-1} + precision^{-1}} = 2 \cdot \frac{precision \cdot recall}{precision + recall} = \frac{tp}{tp + \frac{1}{2}(fp + fn)} $$

**macor-F1**：

适用环境：多分类问题，不受数据不平衡影响，容易受到识别性高（高recall、高precision）的类别影响。

计算每个类别的：$F_1-score_i=2\frac{Recall_i \cdot Precision_i}{Recall_i + Precision_i}$

计算 $macro-F_1=\frac{F_1-score_1+F_1-socre_2+F_1-score_3}{3}$

### 1.3 推荐Baseline

TensorFlow - LongFormer - Baseline: [TensorFlow - LongFormer - NER - [CV 0.633] | Kaggle](https://www.kaggle.com/code/cdeotte/tensorflow-longformer-ner-cv-0-633/notebook)

PyTorch - BigBird - Baseline: [PyTorch - BigBird - NER - [CV 0.615] | Kaggle](https://www.kaggle.com/code/cdeotte/pytorch-bigbird-ner-cv-0-615/notebook)

HuggingFace baseline：[Feedback Prize HuggingFace Baseline: Training | Kaggle](https://www.kaggle.com/code/thedrcat/feedback-prize-huggingface-baseline-training/notebook)

## 2 NER 命名实体识别

### 2.1 什么是 NER命名实体识别？

命名实体识别（Named Entity Recognition，简称NER），又称作“专名识别”，是指识别文
本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。简单的讲，就是
识别自然文本中的实体指称的边界和类别（本次比赛的类型就是 NER 命名实体识别）。

NER是NLP中一项基础性关键任务。从自然语言处理的流程来看，NER可以看作词法分析中
未登录词识别的一种，是未登录词中数量最多、识别难度最大、对分词效果影响最大问题。
同时NER也是关系抽取、事件抽取、知识图谱、机器翻译、问答系统等诸多NLP任务的基础。

### 2.2 NER通常的难点

实体命名识别语料较小，容易产生过拟合；命名实体识别更侧重高召回率，但在信息检索
领域，高准确率更重要；通用的识别多种类型的命名实体的系统性能很差。

### 2.3 NER模型演化路径

### 2.3.1 传统机器学习⽅法：CRF

条件随机场（Conditional Random Field，CRF）是NER⽬前的主流模型。它的⽬标函数不仅考虑输⼊的状态特征函数，⽽且还包含了标签转移特征函数。在已知模型时，给输⼊序列求预测输出序列即求使⽬标函数最⼤化的最优序列，是⼀个动态规划问题，可以使⽤Viterbi算法解码来得到最优标签序列。

其优点在于其为⼀个位置进⾏标注的过程中可以利⽤丰富的内部及上下⽂特征信息。

![crf模型](/posts/kaggle/feedback-prize/crf%20model.png)

### 2.3.2 LSTM + CRF

随着深度学习的发展，学术界提出了Deep Learning + CRF模型做序列标注。在神经⽹络的输出层接⼊CRF层（重点是利⽤标签转移概率）来做句⼦级别的标签预测，使得标注过程不再是对各个token独⽴分类。

![LSTM+CRF模型](/posts/kaggle/feedback-prize/LSTM+CRF%20model.png)

### 2.3.3 BERT + (LSTM) + CRF

BERT中蕴含了大量的通用知识，利用预训练好的BERT模型，再用少量的标注数据进行FINETUNE是一种快速的获得效果不错的NER的方法。

![BERT+LSTM+CRF模型](/posts/kaggle/feedback-prize/BERT+LSTM+CRF%20model.png)

## 3 NLP数据增强

### 3.1 为什么数据增强很重要？

数据增强时通过形成新的和不同的样本来训练数据集，数据增强对于提高机器学习模型的性能和结果是有用的。因为如果机器学习模型中的数据集是丰富和充分的，那么该模型的表现会更好，更准确。

对于机器学习模型来说，收集和标记数据可能是一个耗费精力和成本的过程。通过使用数据增强技术对数据集进行扩增，对于公司/组织可以使减少这些运营成本。

### 3.2 Sentence Shuffling

在 Sentence Shuffling 中，我们将随机洗牌文本中的句子。

**Original Text**：

- Modern humans today are always on their phone. They are always on their phone more than 5 hours a day no stop. All they do is text back and forward and just have group Chats on social media. They even do it while driving.

**Augmented Text**：

- They even do it while driving. They are always on their phone more than 5 hours a day no stop. All they do is text back and forward and just have group Chats on social media. Modern humans today are always on their phone

### 3.3 Remove Duplicate Sentences

在Remove Duplicates中，我们将删除文本中的重复句子。

为了证明这一点，我们把一个句子与它本身连接起来，输出的应该只是原始文本。

**Original Text**：

- Modern humans today are always on their phone. Modern humans today are always on their phone.

**Augmented Text**：

- Modern humans today are always on their phone.

### 3.4 Remove Numbers

在Remove Numbers中，我们将从文本中移除任何数字。我们可以简单地使用正则表达式实现这一点。

**Original Text**：

- There are 15594 samples of training data.

**Augmented Text**：

- There are samples of training data.

### 3.5 Remove Hashtags

在Remove Hashtags中，我们将删除文本中的任何hashtag。

**Original Text**：

- Kaggle Competitions are fun. #MachineLearning

**Augmented Text**：

- Kaggle Competitions are fun.

### 3.6 Remove Mentions

在这个转换中，我们删除了文本中的任何提及（以'@'开头的词）。

**Original Text**：

- @AnthonyGoldbloom is the founder of Kaggle.

**Augmented Text**：

- is the founder of Kaggle.

### 3.7 Remove URLs

在这种转换中，我们从文本中删除任何URL。

**Original Text**：

- https://www.kaggle.com hosts the world's best Machine Learning Hackathons.

**Augmented Text**：

- hosts the world's best Machine Learning Hackathons

### 3.8 Cut Out Words

在这种转换中，我们从文本中删除一些词。

**Original Text**：

- Competition objective is to analyze argumentative writing elements from students grade 6-12.

**Augmented Text**：

- Competition objective is to analyze argumentative elements from grade.

### 3.9 KeyboardAug

KeyboardAug，它使用键盘上的键的相邻性来模拟打错。

**Original Text**：

- Dear Senator, I favor keeping the Electoral College in the arguement. **One of the reasons I feel that way is that it is harder for someone who is runnig for president to win.** To win they would need to win over the votes of most of the **small** states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

**Keyboard augmentation**：

- Dear Senator, I favor keeping the Electoral College in the arguement. **One of the reasons I Vee? that way is that it is harder for soJdoJe who is runnig for president to win**. To win they would need to win over the votes of most of the **s,apl** states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

### 3.10 SpellingAug

KeyboardAug创造的错别字太不自然，接着尝试SpellingAug，它使用一个常见的拼写错误数据库。

**Original Text**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win over the votes of most of the small states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

**SpellingAug augmentation**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win **overt** the votes of most of the **smol** states. Or win over the votes over some of the small states and some of the big states. **To** it would need someone who is smart **onr** at least somewhat smart.

### 3.11 SynonymAug

SynonymAug，它用同义词替换了一些单词。

**Original Text**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win over the votes of most of the small states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

**SynonymAug augmentation**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win over the votes of most of the small states. Or win over the **ballot** over some of the small states and some of the big **res publica. So it would need someone who is
smart or at least somewhat smart**.

### 3.12 WordEmbsAug

使用单词嵌入来寻找类似的单词进行扩增，这里使用的是GloVe模型的本地副本。

**Original Text**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win over the votes of most of the small states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

**WordEmbsAug augmentation**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for **mubarak** to win. To win they would need to win over the votes of most of the small states. Or win over the votes over some of the small states and some of the **bigger** states. So it would need someone who is smart or at least somewhat smart.

### 3.13 ContextualWordEmbsAug

与WordEmbsAug相似，但使用更强大的上下文词嵌入。这里与BERT一起使用。

**Original Text**：

- Dear Senator, I favor keeping the Electoral College in the arguement. One of the reasons I feel that way is that it is harder for someone who is runnig for president to win. To win they would need to win over the votes of most of the small states. Or win over the votes over some of the small states and some of the big states. So it would need someone who is smart or at least somewhat smart.

**ContextualWordEmbsAug augmentation**：

- Dear Senator, I favor keeping the Electoral College in the arguement. **Part** of the reasons I feel that way is that it is harder for someone who is runnig for president **so** win. To win they would need to win over the votes of most of the small **counties**. Or win over the votes over some **among** the small states and some of the big states. So it would need **loser** who is smart or at least somewhat smart.

### 3.14 结论

SpellingAug 和 Contextual WordEmbsAug 看起来产生了很好的结果，可以在不
调整训练时给出的话语注释的情况下使用。

## 4 模型选择

### 4.1 Transformer

Transformer是2017年提出的一种模型架构（《Attention is All You Need》），其优点除了效果好之外，由于encoder端是并行计算的，训练的时间也被大大缩短了。其开创性的思想，颠覆了以往序列建模和RNN划等号的思路，被广泛应用于NLP的各个领域。目前在NLP各业务全面开花的语言模型如GPT, BERT等，都是基于Transformer模型。

Transformer 模型使用了 Self-Attention 机制， Self-Attention 也是 Transformer 最核心的思想，不采用RNN顺序结构，使得模型可以并行化训练，而且能够拥有全局信息。

其中，attention的计算方式有多种，加性attention、点积attention，还有带参数的计算方式。具体可以去看相关文章。

对self-attention来说，它跟每一个input vector都做attention，所以没有考虑到input sequence的顺序。

### 4.2 Bert

BERT是基于transformer的双向编码表示，它是一个预训练模型，模型训练时的两个任务是预测句子中被掩盖的词以及判断输入的两个句子是不是上下句。在预训练好的BERT模型后面根据特定任务加上相应的网络，可以完成NLP的下游任务，比如文本分类、机器翻译等。

虽然BERT是基于transformer的，但是它只使用了transformer的encoder部分，它的整体框架是由多层transformer的encoder堆叠而成的。

每一层的encoder则是由一层muti-head-attention和一层feed-forword组成，大的模型有24层，每层16个attention，小的模型12层，每层12个attention。每个attention的主要作用是通过目标词与句子中的所有词汇的相关度，对目标词重新编码。所以每个attention的计算包括三个步骤：计算词之间的相关度，对相关度归一化，通过相关度和所有词的编码进行加权求和获取目标词的编码。

在BERT中，输入的向量是由三种不同的embedding求和而成，分别是：

1. wordpiece embedding：单词本身的向量表示。WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。

2. position embedding：将单词的位置信息编码成特征向量。因为我们的网络结构没有RNN 或者LSTM，因此我们无法得到序列的位置信息，所以需要构建一个position embedding。构建position embedding有两种方法：BERT是初始化一个position embedding，然后通过训练将其学出来；而Transformer是通过制定规则来构建一个position embedding。

3. segment embedding：BERT 能够处理对输入句子对的分类任务。这类任务就像判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入到模型中。那BERT如何去区分一个句子对中的两个句子呢？答案就是segment embeddings。

BERT的优点是只有BERT表征会基于所有层中的左右两侧语境。BERT能做到这一点得益于Transformer中Attention机制将任意位置的两个单词的距离转换成了1。

### 4.3 Longformer

Longformer 是一种可高效处理长文本的模型，出自 AllenAI 2020年。目前已开源，而且可以通过 huggingface 快速使用。

传统Transformer-based模型在处理长文本时存在一些问题，因为它们均采用“我全都要看”型的attention机制，即每一个token都要与其他所有token进行交互，无论是空间还是时间复杂度都高达O(n^2) 。为了解决这个问题，之前有些工作是将长文本切分为若干个较短的Text Segment，然后逐个处理，例如Transformer-XL。但这会导致不同的Text Segment之间无法进行交互，因而必然存在大量的information loss（信息丢失）。

本文提出的 Longformer，改进了Transformer传统的self-attention机制。具体来说，每一个token只对固定窗口大小附近的token进行 local attention（局部注意力）。并且 Longformer 针对具体任务，在原有 local attention 的基础上增加了一种 global attention（全局注意力）。

Longformer 在两个字符级语言建模任务上都取得了SOTA的效果。并且作者用 Longformer 的attention方法继续预训练 RoBERTa，训练得到的语言模型在多个长文档任务上进行fine-tune后，性能全面超越 RoBERTa

作者共提出了三种新的attention机制，这三种方法都很好的降低了传统self-attention的复杂度，它们分别是滑窗机制、空洞滑窗机制、融合全局信息的滑窗机制。

### 4.3.1 滑窗机制 (SLIDING WINDOW ATTENTION)

对于每一个token，只对其附近的w个token计算attention，复杂度为O(n×w) ，其中n为文本的长度。作者认为，根据应用任务的不同，可以对Transformer每一层施以不同的窗口大小w。

作者在具体实现的时候，设置的窗口大小=512，与BERT的input限制完全一样，所以大家不要存有“ Longformer 比 BERT 更轻量”的错觉。

### 4.3.2 空洞滑窗机制 (DILATED SLIDING WINDOW)

对每一个token进行编码时，普通的滑窗机制只能考虑到长度为w的上下文。作者进一步提出空洞滑窗机制（实际上空洞滑窗是CV领域中很早就有的一项技术），在不增加计算负荷的前提下，拓宽视野范围。在滑动窗口中，被attented到的两个相邻token之间会存在大小为d的间隙，因此每个token的视野范围可达到d×w。实验表明，由于考虑了更加全面的上下文信息，空洞滑窗机制比普通的滑窗机制表现更佳。

### 4.3.3 融合全局信息的滑窗机制 (GLOBAL+SLIDING WINDOW)

我们知道BERT类的语言模型在fine-tune时，实现方式略有不同。比如，对于文本分类任务，我们会在整个输入的前面加上[CLS]这个token；而对于QA任务，我们则会将问题与文本进行拼接后进行输入。在Longformer中，作者也希望能够根据具体任务的不同，在原本local attention的基础上添加少量的global attention。例如，在分类任务中会在[CLS]初添加一个global attention（对应下图第一行第一列全绿）；而在QA任务上会对question中的所有token添加global attention。如下图所示，对于添加了global attention的token，我们对其编码时要对整个序列做attention，并且编码其它token时，也都要attend到它。

### 4.3.4 结论

作者在text8和enwiki8两个字符级任务上对 Longformer 进行了实验。实验中每一层采用了不同的窗口大小，具体来说：底层使用较小的滑窗，以构建局部信息；越上层滑窗越大，以扩大感受野。

训练时，理想状况当时是希望使用GPU所能承受最大的w和sequence length，但为了加快训练速度，作者采用的是多阶段训练法：从较短的序列长度和窗口大小开始，后续每个阶段将窗口大小和训练长度增加一倍，并将学习率减半。

作者一共训练了5个阶段，第一个阶段sequence length是2048，最后一个阶段是23040。

实验结果，Longformer在这两个数据集上皆达到了SOTA。

## 5 HuggingFace

Huggingface Transformer 能够帮我们跟踪流⾏的新模型，并且提供统⼀的代码⻛格来使⽤BERT、XLNet和GPT等等各种不同的模型。⽽且它有⼀个模型仓库，所有常⻅的预训练模型和不同任务上fine-tuning的模型都可以在这⾥⽅便的下载（解决了各种Pretraining的Transformer模型实现不同、对比麻烦的问题）。

设计原则：

- 简洁，只有configuration，models和tokenizer三个主要类。
- 所有的模型都可以通过统一的from_pretrained()函数来实现加载，transformers会处理下载、缓存和其它所有加载模型相关的细节。而所有这些模型都统一在Hugging Face Models管理。
- 基于上面的三个类，提供更上层的pipeline和Trainer/TFTrainer，从而用更少的代码实现模型的预测和微调。
- 它不是一个基础的神经网络库来一步一步构造Transformer，而是把常见的Transformer模型封装成一个building block，我们可以方便的在PyTorch或者TensorFlow里使用它。

主要概念：

- Model类（如 BertModel）：包括30+的PyTorch模型(torch.nn.Module)和对应的TensorFlow模型(tf.keras.Model)。
- Congif类（如 BertConfig）：它保存了模型的相关(超)参数。我们通常不需要自己来构造它。如果我们不需要进行模型的修改，那么创建模型时会自动使用对于的配置。
- Tokenizer类（如 BertTokenizer）：它保存了词典等信息并且实现了把字符串变成ID序列的功能。

### 5.1.1 Pipeline

Pipeline（使用预训练模型的函数），支持如下任务：

- 情感分析(Sentiment analysis)：一段文本是正面还是负面的情感倾向
- 文本生成(Text generation)：给定一段文本，让模型补充后面的内容
- 命名实体识别(Name entity recognition)：识别文字中出现的人名地名的命名实体
- 问答(Question answering)：给定一段文本以及针对它的一个问题，从文本中抽取答案
- 填词(Filling masked text)：把一段文字的某些部分mask住，然后让模型填空
- 摘要(Summarization)：根据一段长文本中生成简短的摘要
- 翻译(Translation)：把一种语言的文字翻译成另一种语言
- 特征提取(Feature extraction)：把一段文字用一个向量来表示

情感分析的例子：

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis’)

results = classifier(["We are very happy to show you the Transformers library.",
                        "We hope you don't hate it."])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# 运行结果
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

如果觉得模型不合适，寻找合适的模型：

```python
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```

### 5.1.2 Tokenizer

Tokenizer的作用大致就是分词，然后把词变成的整数ID，当然有些模型会使用subword。但是不管怎么样，最终的目的是把一段文本变成ID的序列。当然它也必须能够反过来把ID序列变成文本。

```python
inputs = tokenizer("We are very happy to show you the Transformers library.")
```

Tokenizer对象是callable，因此可以直接传入一个字符串，返回一个dict。最主要的是ID的list，同时也会返回attention mask：

```python
print(inputs)

{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 
1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

我们也可以一次传入一个batch的字符串，这样便于批量处理。这时我们需要指定padding
为True并且设置最大的长度：

```python
pt_batch = tokenizer(
    ["We are very happy to show you the Transformers library.", 
    "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

- truncation为True会把过长的输入切掉，从而保证所有的句子都是相同长度的。
- return_tensors=”pt” 表示返回的是 PyTorch的Tensor，如果使用TensorFlow则需要设置。
- return_tensors=”tf”。

分词结果：

```python
>>> for key, value in pt_batch.items():
... print(f"{key}: {value.numpy().tolist()}")
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 
1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 0, 0, 0]]
```

pt_batch仍然是一个dict，input_ids是一个batch的ID序列，我们可以看到第二个字符串较短，所以它被padding成和第一个一样长。如果某个句子的长度超过max_length，也会被
切掉多余的部分。

### 5.1.3 Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Tokenizer的处理结果可以输入给模型，对于PyTorch则需要使用**来展开参数：

```python
# PyTorch
pt_outputs = pt_model(**pt_batch)
```

Transformers的所有输出都是tuple，即使只有一个结果也会是长度为1的tuple：

```python
>>> print(pt_outputs)
(tensor([[-4.0833, 4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>),)
```

### 5.1.4 Config

如果你想自定义模型(这里指的是调整模型的超参数，比如网络的层数，每层的attention head个数等等，如果你要实现一个全新的模型，那就不能用这里的方法了)，那么你需要构造配置类。

每个模型都有对应的配置类，比如DistilBertConfig。你可以通过它来指定隐单元的个数，dropout等等。如果你修改了核心的超参数(比如隐单元的个数)，那么就不能使用
 from_pretrained加载预训练的模型了，这时你必须从头开始训练模型。当然Tokenizer一般还是可以复用的。

下面的代码修改了核心的超参数，构造了Tokenizer和模型对象：

```python
from transformers import DistilBertConfig, DistilBertTokenizer, 
DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)
```

### 5.1.5 NER 命名实体识别

本次比赛是NER类型，我们来HuggingFace在NER上的用法：

```python
from transformers import pipeline
nlp = pipeline("ner")
sequence = ["Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very",
            "close to the Manhattan Bridge which is visible from the window."]
```

```python
>>> print(nlp(sequence))
[
    {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
    {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
    {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
    {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
    {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
    {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
    {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
    {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
    {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
    {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
    {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
    {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
]
```

### 5.1.6 训练

Huggingface Transformers提供了Trainer用作训练：

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

training_args = TrainingArguments(
    output_dir='./results', # output directory
    num_train_epochs=3, # total # of training epochs
    per_device_train_batch_size=16, # batch size per device during training
    per_device_eval_batch_size=64, # batch size for evaluation
    warmup_steps=500, # number of warmup steps for learning rate scheduler
    weight_decay=0.01, # strength of weight decay
    logging_dir='./logs', # directory for storing logs
)
trainer = Trainer(
    model=model, # the instantiated Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=train_dataset, # training dataset
    eval_dataset=test_dataset # evaluation dataset
)
```

TrainingArguments参数指定了训练的设置：输出目录、总的epochs、训练的batch_size、预测的batch_size、warmup的step数、weight_decay和log目录。然后使用trainer.train()和trainer.evaluate()函数就可以进行训练和验证。

我们也可以自己实现模型，但是要求它的forward返回的第一个参数是loss。

## 6 模型融合

### 6.1 融合方法介绍

#### 6.1.1 Voting

Voting可以说是一种最为简单的模型融合方式。假如对于一个二分类模型，有3个基础模型，那么就采取投票的方式，投票多者为最终的分类。

#### 6.1.2 Bagging

Bagging的思想是利用抽样生成不同的训练集，进而训练不同的模型，将这些模型的输出结果综合（投票或平均的方式）得到最终的结果。

其本质是利用了模型的多样性，改善算法整体的效果。Bagging的重点在于不同训练集的生成，这里使用了一种名为Bootstrap的方法，即有放回的重复随机抽样，从而生成不同的数据集。

![bagging](/posts/kaggle/feedback-prize/bagging%20model.png)

#### 6.1.3 Boosting

Boosting是一种提升算法，其思想是在算法迭代过程中，每次迭代构建新的分类器，重点关注被之前分类器分类错误的样本，如此迭代，最终加权平均所有分类器的结果，从而提升分类精度。

与Bagging相比来说最大的区别就是Boosting是串行的，而Bagging中所有的分类器是可以同时生成的（分类器之间无关系），而Boosting中则必须先生成第一个分类器，然后依次往后进行。核心思想是通过改变训练集进行有针对性的学习，通过每次更新迭代，增加错误样本的权重，减小正确样本的权重。知错就改，逐渐变好。典型应用为：Adaboost、GBDT和Xgboost。

#### 6.1.4 Blending

类似概率voting，用不相交的数据训练不同的Base Model，将它们的输出（概率值）取加权平均（两列数据偏差越小，且方差越大，则blend越有效果）。

### 6.2 模型融合 Stacking

交叉验证部分：首先将训练数据分为 5 份，接下来一共 5 个迭代，每次迭代时，将 4 份数据作为 Training Set 对每个 Base Model 进行训练，然后在剩下一份 Hold-out Set 上进行预测。**同时也要将其在测试数据上的预测保存下来**，对测试数据的全部做出预测。

5 个迭代都完成以后我们就获得了一个 **训练数据行数 * Base Model数量** 的矩阵，这个矩阵接下来作为第二层的 Model 的训练数据，训练完以后，对test data做预测。

这时我们得到了两个预测矩阵，平均后就得到最后的输出。

![stacking](/posts/kaggle/feedback-prize/stacking.png)

## 7 总的解决方案思路

采用了 **longformer+ deberta** 的双模型融合，由于官方数据的有一些不干净的原标签，所以我们使用经过修复的corrected_train.csv。

在文本数据的处理上，我们将max_len设置在了1024（在推理是扩大至longformer=4096/deberta=2048）。之后我们对数据做了10Fold的标准切分。

模型上我们选择了 **allenai/longformer-base-4096和microsoft/deberta-large** 版本，在之后接了一个Dropout层和Linear层。

在模型预测出结果后，我们使用了后处理的方式来进一步筛选预测的实体，主要是对每种实体的最小长度和最小置信度做出限制，如果小于阈值则被后处理筛掉。

### 模型代码

```python
class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        # 载入 backbone
        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels) # 分类头

    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        # 五个不同的dropout结果
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5 # 五层取平均
        return logits
```

### 后处理参数

```python
# 每种实体的的最小长度阈值，小于阈值不识别
MIN_THRESH = {
    "I-Lead": 11,
    "I-Position": 7,
    "I-Evidence": 12,
    "I-Claim": 1,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

# 每种实体的的最小置信度，小于阈值不识别
PROB_THRESH = {
    "I-Lead": 0.687,
    "I-Position": 0.537,
    "I-Evidence": 0.637,
    "I-Claim": 0.537,
    "I-Concluding Statement": 0.687,
    "I-Counterclaim": 0.37,
    "I-Rebuttal": 0.537,
}
```

比赛上分历程：

1. longformer Baseline 5Fold，Public LB : 0.678；
2. 将推理阶段的max_len设置为4096，Public LB : 0.688；
3. 加入后处理，Public LB : 0.694；
4. 尝试了deberta-base 但分数太低，我们没有尝试将其加入融合；
5. deberta-large 5Fold 加入后处理，Public LB : 0.705；
6. 将两个模型融合，Public LB：0.709；
7. 对学习率，epoch等进行调参，Public LB：0.712；
8. 使用修复标签后的corrected_train.csv，Public LB：0.714；
9. 尝试将5fold换成10fold，Public LB：0.716；
10. 对后处理进行调参，Public LB：0.718；

## 总结

竞赛是由乔治亚州立大学举办的，对学生写作中的论证和修辞元素进行识别。本次竞赛在数据上我们修复了官方数据的不干净的原标签部分，整体的方案上采用了 **longformer + deberta** 的双模型融合，为了防止过拟合我们尝试了在模型头部位置加入Dropout层。我们训练出了longformer-base-4096和deberta-large两个模型，再通过后处理对每种实体的最小长度和最小置信度做出限制，筛掉小于阈值的预测值，最后进行CV-10Fold和简单的加权融合。此外，我们还尝试了deberta-base等，但没有起效果。最终我们获得了Private LB: 0.718 (Top2%) 的成绩。

