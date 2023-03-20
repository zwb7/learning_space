# Motivation

现有对shallow domain adaption已经有全面的调查，但很少回顾新兴的基于深度学习的方法。

# Contribution

- 根据数据的属性给出了不同Deep DA场景的分类，这些数据定义来了两个域是如何发散的。
- 扩展他人的工作，将基于train loss的方法归纳分类。
- 考虑s和t的distance，研究multi-step，并将其分类。
- 综述了CV方面的应用，突出了当前方法的不足和未来方向。



# Introduction

常见shallowDA：

- instance-based，样本重新加权减少差异
- feature-based，学习一个公共的分布空间，两个数据集的分布是匹配的。

DNN可以学习到更多的可迁移的表示，这些表示可以根据data sample和group feature与不变因子的相关性分层地分离data sample和group feature的探索性因素。但是domain shift仍会影响性能，迁移性在高层急剧降低。**DNN在浅层的迁移性好，但是在更深层的迁移性迅速变差**。

然后介绍了一些人的调查工作，主要集中在浅层的DA和深度DA在图像分类中的应用。

本研究的主要贡献如下：( 1 )根据数据的属性，我们给出了不同深度DA场景的分类，这些数据定义了两个域是如何发散的。( 2 )扩展了Csurka的工作，对(使用分类损失、差异损失和对抗损失进行训练)的3个子集进行了改进和细化，总结了不同DA场景下使用的不同方法。( 3 )考虑源域和目标域的距离，研究多步DA方法，并将其分为hand-crafted、feature-based和representation-based的机制。( 4 )综述了计算机视觉在图像分类、人脸识别、风格转换、目标检测、语义分割和行人重识别等方面的应用

然后给出文章的结构。

# Overview

基于不同的domain divergences将DA分为homogeneous和heterogeneous DA。

homogeneous DA的source和target的特征空间和数据维度是相同的，但数据分布是不同的。( ds = dt)，( X s = X t)，(P(X)s != P(X)t)。

heterogeneous DA的source和target的特征空间是不同的，维度也可能一般不同。

( X s ！= X t)，(ds ！= dt)。

homogeneous和heterogeneous都分为无监督，半监督，有监督。

![image-20230317161112894](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230317161112894.png)

![image-20230317161310558](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230317161310558.png)

上述DA假设source与target直接相关，知识迁移可以一步完成，称为one-step DA

![image-20230317161433480](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230317161433480.png)

Multi-step DA通过一些中间域连接两个域比如人脸图像和车辆图像差异大，通过引入足球头盔图像实现平滑的知识迁移。

## One-step

主要关注adversarial-based，在这种情况下，一个对数据点是否来自源域或目标域进行分类的域判别器被用来通过一个对抗目标来鼓励域混淆，以最小化经验源和目标映射分布之间的距离。

adversiarial-based DA根据是否存在生成模型可以分为两种情况。

generative modals：基于GAN将判别器与生成组件结合。典型的情况是使用源图像、噪声向量或者是两者结合生成与目标样本相似的模拟样本，保留源域的标注信息。

non-generative modals：特征提取器利用source中的label学习一个判别行表示，通过域混淆损失将target data映射到同一空间，从而得到域不变表示。

### Homogeneous domain adaptation

### Adversarial-based approaches

GAN网络以mini-max不等式的方式对label prediction loss进行训练（同时优化生成器以最小化损失，同时训练判别器最大化分配正确label的概率）。在DA中这一原则确保网络无法区分s和t。ADDA框架根据是否使用生成器，使用哪种loss或是否跨域共享权重对现有方法进行总结。

本文仅分为生成模型和非生成模型两个子集。

#### Generative models

带有ground-truth的target data是解决训练数据缺乏的好的选择。在s data的帮助下，生成器生成大量合成t data，这些合成t data和合成s data匹配以共享label。然后使用生成的数据t模型仿佛不需要DA。

![image-20230318211340921](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230318211340921.png)

##### Coupled generative adversarial networks

![image-20230319171131056](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230319171131056.png)

CoGan核心思想是生成与synthetic s数据配对的synthetic t数据。它由一对GAN组成，用于生成s data的GAN1和用于生成t data的GAN2。GAN1前几层和GAN2后几层的权重是绑定的，这使得CoGAN可以在无监督的情况下实现域不变特征空间。训练好的CoGAN可以将输入的噪声向量适应于来自两个分布的成对图像，并共享labels。因此可以利用synthetic t样本的共享label来训练t model。

CoGAN是一种学习源域和目标域数据联合分布的网络，其由两个GAN网络组成。通过强制性的给两个网络加上共享参数的约束，限制了网络的能力，从而两个网络从噪声生成的数据是差不多的。

文章认为生成网络的前几层是解码的高阶语义，所以通过共享权重对齐。判别网络后几层是解码的高阶语义，所以也共享权重对齐。这种权重的约束可以让网络在没有监督的情况下，学习源域和目标域图片的联合概率。所以该网络能用来生成成对的共享高阶语义，而低阶语义不同的图片，例如相同的物体，背景不同的图片。

##### Pixel-level domain transfer

作者设计一个转换网络Converter，将源域的图片 Is 转换成目标域的数据 It^ 。训练过程中， Is 与 It^ 是一一对应的。这边Converter是一个编码解码的过程，但是损失函数不是均方误差MSE，这是因为均方误差有两个缺点：一、MSE会使图片变模糊。二、MSE对于同一个物体例如衣服的各种各样的形状没有容忍度，也就是同样的衣服，只要几何上有一点漂移，就会导致误差很大。这显然不是我们想看到的。

作者受GAN的启发，将损失函数也设计为一个判别网络 Dr，判断生成的图片是否和目标域的一致。但是只有一个判别网络 Dr 的问题是，生成的图片虽然可以和目标域的一致，但是不能保证生成的图片与源域的有关系。所以作者又设计了另外一个判别网络 Da ，输入是源域数据 Is，目标域与之相关的数据 It，目标域与之不相关的数据 It−以及生成的数据 It^ 。判别网络要使 Is与 It 输入时，输出值尽量大， Is 与It−、It^输入时，输出值尽量小，其实也是一个GAN。所以生成网络就需要欺骗 、Da、Dr 两个判别网络。

更多的工作集中在生成与t data相似的synthetic data的同时保持注释。与其他仅将生成器限定在噪声向量或s images上的work不同，有人提出了一个利用同时限定在噪声向量和s images上的GAN模型。

##### Learning from simulated and unsupervised images through adversarial training

提出S+U学习，通过最小化一个对抗损失和自正则损失，来提高合成图片的真实性。用来训练的真实图片不需要标签，修正后的合成数据有标签。这个方法也适用于将源域数据变成目标域，真实数据对应的就是没有标签的目标域数据，修正的合成数据对应于基于源域的修正数据，带有标签，所以S+U还要保证训练的时候不改变数据的标签。

##### Unsupervised pixel-level domain adaptation with generative adversarial networks

这篇文章和Pixel-level domain transfer方法很相似，通过一个生成器，将源域数据 Xs变成目标域的数据 Xf ，用一个判别器来判别真假，当判别器无法判别时，我们认为生成的数据Xf 就和真实目标域 Xt 差不多了。由于生成器并不会改变数据的标签，所以可以用 Xf 与其自带的标签，训练一个判别器用来判别 Xt 。

![image-20230319171953390](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230319171953390.png)



#### Non-generative models

Deep DA的关键是从s和t样本中学习领域不变的表示。有了这些表示两个域的分布可以足够相似，这样即使在源样本上训练，分类器也被欺骗，可以直接在目标域中使用。因此，表征是否为领域混淆对知识转移至关重要。受GAN的启发，引入判别器产生的域混淆损失来提高无生成器深度DA的性能。

##### Unsupervised domain adaptation by backpropagation

![image-20230319163240495](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230319163240495.png)

##### Adversarial discriminative domain adaptation

这篇文章提出ADDA方法，提出一个同样的框架，认为目前基于对抗的域自适应方法都是该通用框架的一个实例。

作者首先与用源域的数据预训练一个分离器。然后再利用GAN的结构将目标域的数据投影到源域。最后将源域的分类器用于目标域的分类。**其实和上篇文章的思路差不多，只不过就是将不同的步骤分开完成，没有耦合在一起**。

 

![image-20230319163257441](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230319163257441.png)

DANN将GRL集成到标准架构中，以确保两个域上的特征分布相似。该网络由共享特征提取层和两个分类器组成。DANN通过使用GRL最小化域混淆损失(对于所有样本)和标签预测损失(对于源样品)，同时最大化域混淆损失。与上述方法不同，ADDA 通过解绑权重考虑独立的源和目标映射，目标模型的参数由预训练的源模型初始化。这更灵活，因为允许更多领域特定的特征提取被学习。ADDA通过迭代最小化以下函数来最小化源和目标表示距离，这与原始GAN最为相似。作者提出增加一个额外的域分类层来执行二进制域分类，并设计了一个域混淆损失来激励它的预测尽可能接近二进制标签上的均匀分布。

##### Simultaneous deep transfer across domains and tasks

将不同标签之间的概率差异作为数据不同类别之间的相关性度量做迁移。中间用到了一个域判别器，用来判别数据来自于源域还是目标域。所以生成器有三个任务，第一需要保证在源域上的分类精度；第二要欺骗域判别器；第三要保存不同的label之间的相关性。

##### Partial transfer learning with selective adversarial networks

这篇文章处理的问题是目标域标签空间只是源域标签空间的子集。比如源域中有足球、望远镜和沙发三个label，而目标域中只有足球和望远镜两个label，可想而之，上面的那些基于对抗的方法包括其他基于统计的方法，效果不会好，因为沙发这个标签的数据会对迁移产生负面的影响(负迁移)。所以作者设计了一个新的对抗机制，来解决这个问题。

选择性对抗网络SAN与以往匹配整个s和t的方式不同，该网络解决了从大域到小域的部分迁移学习。该方法假设目标标签空间是源标签空间的子空间。它同时通过过滤离群源类来避免负迁移，并通过将域判别器拆分为多个类域判别器来匹配共享标签空间中的数据分布来促进正迁移。

##### Few-shot adversarial domain adaptation

FADA将原先的来自于源域还是目标域这个二分类问题，变成了对四组数据的判别。第一组：同一类数据对，均来自于源域；第二组：同一对数据对，分别来自源域和目标域；第三组：不同类数据对，均来自于源域；第四组：不同类数据对，别来自源域和目标域。特征提取器的任务就是要第一组和第二组的分布尽量一致，第三组和第四组的分布尽量一致，让判别器分辨不出来是哪一组的数据。判别器目标相反。

##### Adversarial feature augmentation for unsupervised domain adaptation

用GAN在特征空间做数据增强，同时基于特征增强和域不变设计了一个新的无监督DA方法。

算法分三步：

第一步：使用源域数据训练一个编码器 Es和分类器 C 。

第二步：使用源域编码器 Es 提取特征，并将特征生成器 S与判别器 D1 对抗。得到增强的源域数据和标签。

第三步：域不变特征编码器 Ei 提取源域数据和目标域数据特征，同时用特征生成器生成的特征一起与判别器 D2 对抗。最后用源域生成的分类器 C 对目标域特征进行分类。

##### Wasserstein distance guided representation learning for domain adaptation

这篇文章是把WGAN的度量用在了domain adaptation上，提出WGDRL度量。具体的网络设计和其他方法差不多，就是特征提取，然后类别判别，但是中间的域判别用了Wasserstein距离。

##### Maximum classifier discrepancy for unsupervised domain adaptation

很多对抗学习方法都是找一个域判别器来判断数据来自于目标域还是源域，然后特征提取器来期盼判别器从而在特征空间中匹配源域和目标域的分布。但是这样存在两个问题：

一、域分类器只考虑数据来自哪个域，并没有考虑到数据的决策边界，打个比方，源域数据特征分布是[0, 1]的均匀分布，以0.5为界，[0, 0.5]是正样本，[0.5, 1 ]为负样本。即使目标域特征分布也变为[0, 1]，如果决策边界不一样，效果还是不会好。这个有点类似于源域和目标域在标签空间没匹配，没做到类内方差最小，类间方差最大。

二、这些方法想完全匹配两个域的数据，往往也是行不通的。因为每个域都有自己的独有特性，可能无法完全匹配。

文章就是想解决这两个问题。方法是找两个分类器 ，F1，F2 ，让它们在源域上分类尽量准确，在目标域上预测差异尽量大。

### Reconstruction-based approaches

在DA中，源或目标样本的数据重建是一项辅助任务，同时侧重于在两个域之间创建共享表示，并保留每个域的个体特征。

#### Adversarial reconstruction

##### Unpaired image-to-image translation using cycle-consistent adversarial networks

![image-20230319180217578](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230319180217578.png)

大名鼎鼎的CycleGAN，其实也是用了类似于自编码器的结构，实现了风格迁移，做出了很多有趣的应用。利用了GAN和自编码器的想法，其实也可以放到基于对抗迁移的集合里面。用大白话一下思想：

要将图片 X 风格转换到 Y 风格，比如现实风格变成卡通风格，那么可以用一个生成器 G 生成 Y^ ，然后用一个判别器 Dy 判断真假。其实这边并不要求数据是一一对应的，也就是一个图片 X 训练的时候，没有与其对应的 Y 。那么怎么保证 X 映射过去后内容不变呢？？想法是我再让它映射回来呗，是不是很像个自编码器！而且保持原有结构映射回来，肯定比丢掉结构后变成完全不相关的图片再映射回来要容易很多，显然网络会选择前者。有点奥朗姆剃刀的意思，选简单的那个嘛。所以再设计一个生成器 F ，将 Y 映射回X，用另一个判别器 Dx 判别真假。对 Y 的数据，用同样的 、F、G 映射回 Y 。所以整个网络一共有两个生成器 、G、F ，两个判别器 、Dx、Dy 。

##### Dualgan: Unsupervised dual learning for image-to-image translation

与CycleGAN想法类似的DualGAN，唯一有点差别的是损失用了WGAN的损失。该生成器在镜像下采样和上采样层之间配置了跳跃连接，使其成为一个U型网络来共享底层信息(例如,物体形状、纹理、杂乱等)。对于判别器，采用Markovian patch-GAN架构捕获局部高频信息。

##### Learning to discover cross-domain relations with generative adversarial networks

DiscoGAN，核心思想也是和CycleGAN和dualGAN一样的。在DiscoGAN中，可以使用多种形式的距离函数，如均方误差( MSE )、余弦距离和铰链损失作为重建损失，并将网络应用于翻译图像，在保持所有其他组件的同时，改变包括头发颜色、性别和方向在内的指定属性。

### Heterogeneous Domain Adaptation

异构数据领域自适应：如果源域和目标域数据空间是一致的，也就是 Xs=Xt ，数据维度也是一样的 Ds=Dt ，但是数据分布不一致 P(Xs)≠P(Xt) ，比如都是图片，一个是淘宝上下载的，一个是现实中拍摄的，称为Homogeneous DA。但是如果数据空间也不一致呢？就是 Xs≠Xt ，并且 Ds≠Dt ，比如源域现实中的图片，目标域是卡通图片，甚至源域是图片，目标域是文本也是可能的。显然基于同构的数据自适应方法无法处理这种情况。本文梳理一下都有哪些方法可以处理。

#### Adversarial-based approach

##### Unsupervised cross-domain image generation

作者又提出来一个DTN（Domain Transfer Network）方法，如果有印象，在基于统计差异重构中的Deep Transfer Network缩写重名有木有！用的数据集也都是图片，比如源域S是真实风格的图片，目标域 T 是卡通风格的图片，想做的任务是将真实风格的图片转化成卡通风格的，见下图。

##### Generative adversarial text to image synthesis

上面都是处理的图片到图片到自适应，这篇文章处理由文本生成图片。其实大家看图就能知道怎么做了。生成器先将文本通过某种方式编码到特征空间，比如词向量。然后和GAN的输入噪声合并，一起生成对应的图片。判别器将图片解码到特征空间，判别器中文本到作用与上面那篇 f 作用类似，希望生成器生成的图片，与文本是一一对应，而不是随意的一张图片。

##### StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks

同样处理文本到图片，不过是想生成256x256高清图片。怎么做呢？不妨先生成64x64的图片，再从64x64生成256x256的。其实生成过程与上一篇类似，这里不细说了。

##### High-quality facial photo-sketch synthesis using multi-adversarial networks

基于重构的方法，类似于CycleGAN，处理的是真实图片与素描的转换。创新之处在于用了多个判别器对抗，在图片的不同尺寸上进行生成对抗的过程，这样做的好处是能够生成高清的图像。

## Multi-step

首先确定与source和target更相关的intermediate，不直接连接s和t。其次知识通过one-step DA在s、t、i之间转移。关键是如何选择和利用i。分类：

- Hand-crafted：用户根据经验确定。
- instance-based：从auxiliary datasets中选择i。
- Representation-based：通过freeze之前train的nn并使用他们的中间表示作为新nn的输入，实现传输。

## Application of deep domain adaptation

![image-20230320120058357](C:\Users\msi\AppData\Roaming\Typora\typora-user-images\image-20230320120058357.png)

# Conclusion

从广义上讲，深度DA是利用深度网络来提高DA的性能，例如使用深度网络提取特征的浅层DA方法。从狭义上讲，深度DA是基于深度学习架构进行设计，并通过反向传播进行优化。这篇文章关注狭义的定义。

Deep DA分为homogeneous和heterogeneous，进一步分为有监督，半监督，无监督，以前的工作大多集中在无监督的情况，而半监督更值得关注。

考虑s和t的距离，Deep DA分为一步和多步。

Deep DA尚未解决的问题：

- 现有算法大多关注homogeneous，假设s和t的特征空间和维度相同。而这一假设在很多应用中可能不成立。未来heterogeneous可能更受关注。
- Deep DA的实际应用只有少数论文解决了分类和识别以外的适应问题，如目标检测、人脸识别、语义分割、行人再识别。如何在没有数据量或数据量非常有限的情况下实现这些任务是未来需要面对的挑战。
- Deep DA旨在对其边缘分布，通常假设跨s和t的共享label空间。而现实场景中，s和t的图像可能来自不同的类别集合，或者只共享少数类别，这一问题值得进一步关注。



文章大部分篇幅在介绍one-step DA，multi-step DA着墨不多，是否multi-step DA问题研究较少。

文章大部分介绍homogeneous DA，是否heterogeneous DA更值得关注。

文章总结的工作大部分以分类为主，谈到分割的内容不多，而在医学影像领域分割是很重要的一个人物，是否应该去关注一些DA中分割的内容。

文章是18年发表的，后续至今Deep DA领域的发展需要继续去关注。



2871918022