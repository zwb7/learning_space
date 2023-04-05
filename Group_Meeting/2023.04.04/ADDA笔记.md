# Motivation

1. 在做区分的任务或者域偏移较大任务上之前的方法比较不那么令人满意。例子是：CoGANs。这种方法只在源域和目标域非常相似的情况下(如MNIST和USPS)显示出优势，在我们的实验中，我们很难使它收敛到更大的分布偏移。
2. 在能够处理大偏差任务的方法上，大多采用了固定权重的方法来进行训练（主要就是target domain的分类那边，基本是源和目标共用一个分类器）。
3. 其实很多方法都没有用到GAN的标准的loss。
4. 现在的方法基本都是说，让一个映射，将target映射到source ,然后用source的分类器来处理。但是没有说学习两个映射，将source /target 都映射了，然后用映射后的来进行分类。

# Abstract

生成方法可视化做的很好，但在判别任务上不是最优的，很容易受到小规模domain shift的限制。另外判别方法可以处理更大的domain shift，但会对模型施加绑定权重，并且不利用基于GAN的损失。

模型特点：结合了判别模型，无约束权重共享，GAN loss。

优点：比竞争性的domain adversarial方法更加简单。

# Introduction

CNN由于domain shift和dataset bias的影响，在一个大的数据集上训练的识别模型并不能很好的在新的数据集和任务上泛化。典型的解决方案是在特定任务的数据集上fine-tune网络，但难以获取足够多的标注数据且成本高。

作者发现输入图像分布的生成建模不是必须的，因为最终的任务是学习判别表示。另一方面，非对称映射可能比对称映射更好地模拟低级特征的差异。

ADDA 用在source的label学习一个discriminator，再用一个不同的编码方式，能利用非对称映射通过域对抗损失优化来把target data映射到相同的空间。

# Generalized adversarial adaptation

![image-20230330160403301](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330160403301.png)

![image-20230330160425601](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330160425601.png)

ADDA在source和target的映射间选取的基模型是判别式的，采用无约束权值共享，对抗目标是GAN的loss。

source和target的映射是独立的，要学的只有Mt。

# Method

 论文方法讲了很多，其实就是想解决三个问题：

1. 是使用一个基于生成器还是鉴别器的模型
2. 是使用固定的参数还是不固定的参数（对称的还是非对称的）
3. loss 函数怎么选



# Adversarial discriminative domain adaptation

![image-20230330172820717](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330172820717.png)

***图解：***我们提出的对抗性判别域适应 (ADDA) 方法的概述。我们首先使用标记的源图像示例预训练源编码器 CNN。接下来，我们通过学习目标编码器 CNN 来执行对抗性适应，这样看到编码源和目标示例的鉴别器就无法可靠地预测它们的域标签。在测试期间，目标图像通过目标编码器映射到共享特征空间，并由源分类器进行分类。虚线表示固定的网络参数。

首先：使用含标签的源图像训练源域上的CNN和一个分类器。

然后：通过对抗的方法训练一个用于目标域的CNN和一个判别器，判别输入来自源域还是目标域，从而达到对抗的效果（打个比方：现在有个判断动物是否有尾巴的模型，source是马，target是猪，这个网络就是希望把它们“尾巴”的共同特征找到，而不是把短尾当没有）。

测试中：目标域上的CNN和一开始的分类器就可以用来对目标域上的图像进行分类。虚线表明这是固定的网络参数（意思是直接套用的）。

我们知道，由于目标域没有标签信息，所以我们肯定没有办法在目标域上学习映射 Mt和分类器 Ct ，但是我们可以通过减小源域和目标域映射分布的差异，将源域和目标域的样本映射到同一个特征子空间中，我们在源域上训练到的分类器就可以直接用到目标域上。

关于上述流程第二步：

- 为什么要无约束权值共享？这是一个灵活的学习模式，能学习到更多领域特征。
- 为什么要保留一部分权值？有可能产生退化解。
- 怎么解决？把对source预训练出的模型作为target表达空间初始版本再通过训练去改进。

![image-20230330202918400](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330202918400.png)

损失函数如上，第一个式子训练一个分类器C和用于源域上的CNN模型Mt。

第二个式子训练一个用于区分源域和目标域数据的判别器D。

具体操作就是：

1. 通过源域上的样本和标签训练 Ms 和 C 。
2. 然后我们保持Ms不变，用 Ms 初始化 Mt ，然后通过优化第二项和第三项得到最终的 D 和 Mt 。

最终就可以直接使用上面训练的 Mt 和 C 对目标域上的样本进行分类。

第三部分的各种损失函数：

首先是训练源域上的CNN和分类器，用的就是标准的有监督损失：

![image-20230330203521056](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330203521056.png)

E是数学期望，这里是离散分布所以就是取个均值。1是因为域判别器的输出在0和1之间，源域的标签为1，目标域的标签为0，域判别器希望让目标域数据的输出D(Mt(xt))尽量靠近0，这样LadvD那个损失就能最小化。

然后我们需要减小源域和目标域的距离：

首先训练一个判别器判断一个数据点来自源域还是目标域：

![image-20230330203549193](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330203549193.png)

我们希望最小化该损失，即希望训练的判别器尽可能准确地分辨出输入来自源域还是目标域。

![image-20230330203610723](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330203610723.png)

文中提到，目前所有的方法都是先用 Ms 初始化 Mt 的参数，只不过不同的方法选择不同的约束<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330203654029.png" alt="image-20230330203654029" style="zoom: 50%;" />。

有一种常见的约束就是 Ms 和 Mt 的参数是相同的：

<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214339589.png" alt="image-20230330214339589" style="zoom:50%;" />

<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214402065.png" alt="image-20230330214402065" style="zoom:50%;" />表示Ms的第i层。

文中采取的方式是不加任何约束，即 Ms 和 Mt 是不共享参数的。

下面看训练 Mt 的损失函数：

Unsupervised Domain Adaptation by Backpropagation 中的方法是利用梯度反转层来最大化判别器D的损失 <img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214535039.png" alt="image-20230330214535039" style="zoom: 33%;" />：

<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214604368.png" alt="image-20230330214604368" style="zoom:50%;" />

但是这个损失函数存在问题：训练开始时判别器收敛的很快导致梯度消失。还有就是GAN中的损失函数，相当于有两个独立的损失函数，一个是生成器的，另一个是判别器的，因此  ：

<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214640397.png" alt="image-20230330214640397" style="zoom:50%;" />

这个损失函数虽然有更强的梯度，但是当两个分布都变化时，这个损失函数会导致振荡。

Simultaneous Deep Transfer Across Domains and Tasks中使用的是交叉熵损失：

<img src="C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230330214719448.png" alt="image-20230330214719448" style="zoom:50%;" />

这个方法呢，就综合考虑了两个域相互接近的过程，比较稳定。

介绍了这么多的loss 的优缺点，继续看下面的。但因为我们使用预训练的源模型作为目标表示空间的初始化，并在对抗性训练中修正源模型。这样做，我们有效地学习了一个非对称映射，在这个映射中，我们修改目标模型以匹配源分布。这与原始的生成对抗性学习环境最相似，生成的空间被更新，直到与固定的实际空间无法区分为止。因此，我们选择上一节描述的inverted label GAN loss。
