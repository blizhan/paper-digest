---
type: Research Paper Review
title: "TabPFN-3.5: Technical Report"
description: "TabPFN-3.5 的 per-cell Fourier/ECDF 编码、联合分类回归、合成先验与概率预测结果；结合官方源码讨论 Query 交互及时间关系建模边界。"
resource: https://arxiv.org/abs/2609.17895v1
tags: [tabular-foundation-model, in-context-learning, synthetic-prior, probabilistic-regression, temporal-generalization]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2609.17895v1
    title: "TabPFN-3.5: Technical Report"
  - id: code
    resource: https://github.com/PriorLabs/TabPFN
    title: PriorLabs/TabPFN
  - id: weights
    resource: https://huggingface.co/Prior-Labs/tabpfn_3_5
    title: Prior-Labs/tabpfn_3_5
---

# TabPFN-3.5: Technical Report

## Links

- Paper: [arXiv:2609.17895v1](https://arxiv.org/abs/2609.17895v1) · [HTML](https://arxiv.org/html/2609.17895v1) · [PDF](https://arxiv.org/pdf/2609.17895v1)
- Official code: [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
- Weights and license: [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5)
- 本笔记的公开代码分析基于官方仓库 commit [`eeb37a6c4e4b8803af41faa57c3672c61caa5b44`](https://github.com/PriorLabs/TabPFN/commit/eeb37a6c4e4b8803af41faa57c3672c61caa5b44)。论文、代码和商业 API 分开讨论；没有审计 Plus / Thinking 的内部实现。

## 一句话结论

TabPFN-3.5 没有改写「合成任务预训练 + 真实表格 in-context prediction」的范式，而是同时改进 **per-cell 数值表示、分类/回归共享主干、模型容量、synthetic prior 和缓存推理**，在多种 tabular benchmark 上形成很强的默认模型。特别值得注意的是 ScoringBench 的预测分布质量，以及在 grouped / temporal split 上缩小与调优模型的差距。

我们的判断是：它善于利用**已表达为列的时间信息**，但公开架构没有显式的时间顺序或 Query 间状态传递；ScoringBench 的提升也不能单独归因于 ECDF。论文主榜表格还为不同 benchmark 选择不同家族成员，不能直接解读成同一个开源 checkpoint 在全部任务上第一。

## 文章摘要

TabPFN-3 已能接收最多约百万行的推荐输入，但此前的表格基础模型在高基数类别、宽表、文本、非 IID grouped / temporal split 和大规模数据上仍有缺口。TabPFN-3.5 保留整体列分布编码 → 行内特征聚合 → 跨行 ICL 的架构，通过可学习 Fourier 频率和 in-context ECDF 加强单元格表示，将 ICL 宽度从 512 扩至 1024，并以一个 checkpoint 联合支持分类和回归。合成先验强调高基数、宽表和 train/test 不同组的任务。[论文 §1、§3](https://arxiv.org/html/2609.17895v1#S3)

评估覆盖 TabArena、BeyondArena、STRABLE、MulTaBench、RelArena-α、TALENT、ScoringBench，另有 fev-bench 的时间序列实验。论文还给出 Fast checkpoint，以及通过 API / 企业部署提供的 Plus 与 Thinking。后两者的实现细节被作者明确保留为 proprietary。[论文 §2、§3.4](https://arxiv.org/html/2609.17895v1#S3.SS4)

## 相比 TabPFN-3 到底改了什么

| 维度 | TabPFN-3 | TabPFN-3.5 |
| --- | --- | --- |
| checkpoint | 分类 53M、回归 58M，分别训练 | 分类/回归统一 220M；Fast 为 84M |
| 推荐训练行数 | 1M | 1M |
| 推荐特征数 | 2,000 | 6,000；增加 estimators 时论文称可覆盖约 20,000 |
| 输入编码 | 标准化值、缺失标记、feature grouping | 再加入 learned Fourier features 和基于训练列的 ECDF |
| ICL 表示 | 512 维、8 heads、每行 4 个 CLS | 1024 维、16 heads、每行 8 个 CLS |
| 任务 | 分类、回归各自 checkpoint | 共享 cell/column/row/ICL 主干，分开的标签编码器和输出头 |
| 推理 | 多 estimator + 缓存 | 更新 decoder-key 缓存；ECDF 分桶缓存；另有 Fast 变体 |

这里的行/列数字是**推荐有效范围，不是硬件上限**。220M 增长与性能改善同时发生，不能把实验收益全归因于任何单个新模块。[论文 §1、§3](https://arxiv.org/html/2609.17895v1#S3.SS1)

## 方法拆解

### 1. Fourier 和 ECDF 作用于每个单元格

对单元格 `x[i,j]`，模型先记录缺失/无穷标记，填补并根据训练行进行标准化。标准化数值与可学习频率相乘，再经过 `sin/cos`；ECDF 给出该值相对于**同一列训练行**的分布位置 `u∈[0,1]`，再以少量低频 `sin/cos` 展开。标准化原值仍保留在 metadata 路径中。[论文 §3.1](https://arxiv.org/html/2609.17895v1#S3.SS1)

这两个 Fourier 展开容易混淆：主数值路径使用默认 32 个**可学习**频率；ECDF 路径使用默认 4 个低频。它们都是对标量生成编码维度，并没有沿 `sample axis` 对一列样本做 FFT。训练行和 Query 行经过同一套编码器，推理时使用已预训练的频率参数。Query 的 ECDF 则要读取当前任务的训练列分布。[官方实现 `FourierFeatureGroupEmbedder`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L810)、[`_preprocess_raw`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L2547)

ECDF 利用排序增强对偏态、长尾和单调变换的鲁棒性；它表达的是**输入特征的边际分布位置**，并不直接估计目标变量的条件预测分布。实现使用最多 8192 个 bucket 建立训练列 ECDF context，bucket 边界上的 midrank 精确，超过 bucket 容量时对间隔值插值。因此论文中的单调变换不变性，在有限 bucket 的近似实现里不应无条件理解为数值逐位相同。[官方实现 `_build_ecdf_context` / `_in_context_ecdf`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L3119)

### 2. 三层交互各做什么

```text
一张表的训练行 + Query 行
  ↓  cell encoding（逐单元格；Fourier/ECDF）
同列跨训练行：inducing points 汇总列分布和已知标签
  ↓
同行跨特征：CLS tokens 聚合各列为行表示
  ↓
跨行 ICL：训练行读取训练行，Query 行读取训练行
  ↓
分类 logits / 回归分布 bins
```

第一阶段的 per-column `FeatureDistributionEmbedder` 使用 3 个 induced self-attention blocks 和默认 128 个 inducing points。inducing points 只从训练行聚合信息，训练行与 Query 再读取该汇总。第二阶段 `ColumnAggregator` 在**一行内**对列 token 做 attention，8 个 128 维 CLS 表示拼接成 1024 维行向量。最后 24 层 ICL blocks 在行轴处理上下文。[`FeatureDistributionEmbedder`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1757)、[`ColumnAggregator`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1841)、[`TabPFNV3p5`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L2066)

ICL attention 的 K/V 取自训练行；训练行可看训练行，Query 行可看训练行，Query 不以其他 Query 为 K/V。分类头还有基于训练标签的 many-class decoder；回归头输出固定 bins 的预测分布。这支持「单批 Query 中加入另一个 Query 通常不应改变当前 Query 的模型输出」的架构判断，但数值内核和外围配置仍需在具体调用方式下验证。[`ICLAttention`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1178)、[`MultiTaskHeads`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1966)

### 3. 预处理、合成先验和推理缓存

论文称 3.5 不再依赖旧版本的额外 quantile / robust scaling 和 SVD 特征增强；**这指论文中的 3.5 推理配方**。公开 Python 包仍保留这些可选步骤，实际运行组合由 checkpoint 自带的 `InferenceConfig`、用户覆盖配置和 estimator ensemble 决定；不能看到通用 `pipeline_factory.py` 里有 SVD 类就断言默认 3.5 一定用了 SVD。[论文 §3.2](https://arxiv.org/html/2609.17895v1#S3.SS2)、[`pipeline_factory.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/preprocessing/pipeline_factory.py#L49)、[`InferenceConfig.get_default`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/inference_config.py#L437)

公开 estimator 的 `fit()` 先做日期/文本展开、类别识别与编码，再为各 estimator 拟合训练集预处理；`predict()` 对 Query 复用已拟合的转换器。日期处理扩展日历特征；启用文本处理时，`skrub.StringEncoder` 将字符 n-gram TF-IDF 降到数值维度。这是 OSS 的显式特征工程，不等于 Plus / Thinking 的 proprietary 原生文本能力。[`classifier.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/classifier.py#L790)、[`DateTransformer`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/preprocessing/datetimes.py#L52)、[`TextTransformer`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/preprocessing/text.py#L61)

作者称合成数据生成延续 TabPFN-3，同时增加高基数类别、宽表、train/test 不同 group 的任务，并借鉴 TabICLv2 prior。论文并未给出可独立复核的完整生成器和各项消融；本次读到的 `tabpfn_v3_5.py` 文件明确是 *inference only*，不能将它当成预训练实现。[论文 §3.3](https://arxiv.org/html/2609.17895v1#S3.SS3)、[架构文件头](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1)

缓存路径保存列分布的 inducing states、ICL 的训练行 K/V、标准化统计量与 ECDF buckets；分类任务还保存 many-class decoder 的训练 keys。Query 可单独传入，重用训练上下文。缓存减轻重复计算，但论文说明大于 128k 训练行时整体推理仍呈二次增长。[论文 §2.6](https://arxiv.org/html/2609.17895v1#S2.SS6)、[`forward` 缓存路径](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L2227)

## 关键实验结果与解释边界

- **TabArena**：51 个数据集、816 个 splits。论文称 3.5 超过 TabFM 84 Elo，所用总时间约为其十分之一；相对 AutoGluon 1.6 extreme 高 130 Elo、时间约为五分之一。时间轴采用 fit / 1,000 训练行 + predict / 1,000 测试行的中位数，不是每个场景的端到端 SLA。[§2.1、附录 C.1.4](https://arxiv.org/html/2609.17895v1#S2.SS1)
- **BeyondArena**：142 个数据集、507 个 splits 的 core view。3.5 总体第一，但 tuned + ensembled MLP 在完整 grouped、temporal、大规模子集仍领先；去掉大表（≤100k 行）后，3.5 在 grouped、temporal 子集与最强基线相近。作者提醒 temporal / grouped slice 与数据规模混杂，且在一类 grouped 任务中向 TabPFN 暴露 group identifier。[§2.2、附录 C.2.3](https://arxiv.org/html/2609.17895v1#S2.SS2)
- **ScoringBench**：101 个 OpenML 回归数据集，各抽样至 3,000 行、5 折。评价完整预测分布的 CRPS 平均排名：3.5 为 **2.85**，Fast 为 **5.76**，TabPFN-3 为 **7.64**；3.5 在 101 个数据集中的 85 个上胜过 3，按数据集的中位相对 CRPS 改善 **1.3%**。基线使用 benchmark 作者已发布结果，3.5/Fast 在其 harness 中运行，作者另行重跑 TabPFN-3 检查可比性。该结果不等于已经验证大样本、尾部事件、长期漂移下的校准。[附录 C.7](https://arxiv.org/html/2609.17895v1#A3.SS7)
- **文本 / 图像**：STRABLE 的开源 3.5/Fast 使用 TF-IDF；MulTaBench 的开源 3.5/Fast 使用冻结的文本/图像 embedding 并 PCA 到 30 维。Plus / Thinking 接收 raw text，但图像仍依赖冻结 embedding。比较沿用 benchmark 已发布的 baseline 结果，并未重跑全部 baseline。[§2.3–2.4](https://arxiv.org/html/2609.17895v1#S2.SS3)
- **RelArena-α**：TabPFN-Rel + 3.5-Plus 是 system / API 路径。替换 3-Plus backbone 带来近 90 Elo，另一个**内部**更新的 Rel harness 再提高 36 Elo；公开 checkpoint 和 Thinking 的对应结果当时仍在进行，不能把这些数字当作纯本地 3.5 的提升。[§2.5](https://arxiv.org/html/2609.17895v1#S2.SS5)
- **推理时间**：在单张 NVIDIA RTX PRO 6000 Blackwell 96GB 上，按默认 8 / 8 / 4 estimators 折算，标准 3.5 在大训练集上最高比 3 慢约 2 倍；Fast 最高比标准 3.5 快约 6 倍。API Plus 的 FP8 优化与本地 OSS 实现应分开理解。[§2.6](https://arxiv.org/html/2609.17895v1#S2.SS6)

总览表的「七榜第一」是**每个 benchmark 挑选最佳家族成员**：前四个用 Thinking，RelArena-α 用 TabPFN-Rel (3.5)，TALENT 和 ScoringBench 才是标准 3.5。不能用它证明同一个公开 checkpoint 横扫七榜。[论文 Table 1、附录 C.8](https://arxiv.org/html/2609.17895v1#A3.SS8)

## 关键分析与我们的讨论

### ScoringBench 的优势主要来自 ECDF 吗？

**目前不能归因。** ECDF 有助于表示偏态/长尾输入，但 CRPS 衡量的是 `p(y|X, context)` 的预测分布。Fourier 编码、模型扩容、synthetic prior、多任务训练、回归输出头与 estimator ensemble 都同时变化。论文没有固定其他因素、只去除 ECDF 并重训的 ScoringBench 消融。我们的机制推断是「输入分布表示 + prior + 容量可能共同作用」，不是作者证明的贡献拆分。[§3.1–3.3、附录 C.7](https://arxiv.org/html/2609.17895v1#S3)

### Fourier 是否沿 sample axis 做变换？Query 会被编码吗？

Fourier 在**每个单元格的数值维度**展开：`x[i,j] → [sin(ω₁x[i,j]), cos(ω₁x[i,j]), ...]`。Train 与 Query 共用编码器；它本身不聚合其他样本。真正跨训练样本的计算发生在列分布 embedder 和 ICL attention；ECDF 则通过训练列统计量间接依赖其他样本。公开代码中不同 group position 有各自频率参数，但这些参数在行和列上的同一位置复用。[`FourierFeatureGroupEmbedder`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L810)、[`ICLAttention`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1178)

### 表格样本有时间依赖时，现有 ICL 能利用多少？

要区分「时间已变成列」与「行之间有动态」。calendar、lag、rolling statistics、历史预测值、entity ID 等在 cutoff 前可用的特征可以进入表格，模型能利用它们。相反，公开架构没有按时间位置编码行，也没有 Query→Query attention 或自回归状态递推；仅按时间给训练行排序，不能使模型自动学到相邻时刻的动态。列分布汇总和 ICL 确实跨样本交换信息，但采用训练行集合的方式，不能直接等同于时间序列建模。这是我们依据公开调用链得出的架构判断，不是论文声称「时间任务无效」。[代码：列分布与 ICL](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L1643)

对于时间任务，额外要守住特征构造的 cutoff：每个 `(entity, time)` 样本只能使用当时可获得的历史和未来已知 covariates；ECDF / 标准化也必须用训练上下文拟合。随机切分上的高分不能回答未来时段表现。一个可检验的判断是：固定每行特征，打乱训练行顺序是否应改变预测？当前架构不把原始行次序编码为时间，所以若任务确实依赖「前一个 Query 的状态」，需要显式 lag 构造、滚动更新上下文或序列模型。[§2.2](https://arxiv.org/html/2609.17895v1#S2.SS2)、[`_preprocess_raw`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L2547)

## Insights

1. **输入编码与任务先验要一起看**：ECDF/Fourier 提供数值表示，synthetic prior 决定模型在预训练中见过什么样的表格机制；单看其中一个无法解释整体收益。
2. **概率分布 benchmark 与业务校准不是一回事**：CRPS 平均排名很强，但 tail risk、条件覆盖率、长期时间漂移仍需在任务数据上检验。
3. **Context selection 也是时间任务的模型设计**：对于漂移明显的数据，训练上下文包含哪些历史时段、是否按 recency 选择，可能和 backbone 变化同样重要；公开基础架构没有自动解决这一步。
4. **分清 model 与 system**：开源 3.5、Fast、商业 Plus/Thinking、RelArena harness、fev-bench 的时间特征封装，是不同层级的能力和结果。

## 我们的观点

这是一次很强的 foundation-model 工程与表示迭代，方法价值主要在更可靠的 per-cell 编码、共享多任务主干及大表缓存设计；它为「强表格默认模型」提供了有说服力的广覆盖证据。但没有充分的模块级消融支撑「ECDF 导致概率预测提升」之类的单因果解释，且商业最强结果混合了未公开的 inference-time compute 和系统实现。

对我们关心的概率预测和时间任务，它适合作为严格时间切分下的重要 baseline，尤其值得检验带 lag/rolling 特征的回归分布质量；不能把它视为原生序列状态模型。这个判断来自论文 BeyondArena 的保留结果和公开代码中 train-only K/V、无 Query 间交互的实现。

## GitHub / Code Analysis

官方仓库 commit：`eeb37a6c4e4b8803af41faa57c3672c61caa5b44`。

| 源码 | 我们实际核对的内容 |
| --- | --- |
| [`classifier.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/classifier.py#L790)、[`regressor.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/regressor.py#L925) | `fit()` 清洗、编码、生成 ensemble views；`predict()` 复用拟合的变换器 |
| [`preprocessing/pipeline_factory.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/preprocessing/pipeline_factory.py#L49)、[`inference_config.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/inference_config.py#L437) | 预处理按配置选择；通用包保留旧变换，v3.5 推理配置来自 checkpoint |
| [`tabpfn_v3_5.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/architectures/tabpfn_v3_5.py#L2066) | per-cell Fourier + ECDF、列分布汇总、行内 CLS 聚合、train-only ICL attention、分类/回归输出头 |
| [`inference.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/src/tabpfn/inference.py#L870) | `fit_with_cache` 为每个 estimator 建 train cache，`predict()` 只处理 Query 并复用 context |
| [`test_tabpfn_v3_5.py`](https://github.com/PriorLabs/TabPFN/blob/eeb37a6c4e4b8803af41faa57c3672c61caa5b44/tests/test_architectures/test_tabpfn_v3_5.py#L617) | 测试校验 cached / uncached ECDF 排名和输出一致性；bucket 近似是显式测试场景 |

没有下载大权重或重跑论文 benchmark。本笔记对**模型结构和公开推理路径**的描述来自实际源码；默认 checkpoint 内具体 `PREPROCESS_TRANSFORMS` 值未从权重中解包验证；合成数据生成与 Plus / Thinking 具体策略只能据论文和官方说明记录。

开源 Python 代码采用 Apache-2.0；3.5/Fast 权重另受 TABPFN-3.5 License v1.0 限制，可用于研究及内部评估，生产/商业使用另需授权。Plus/Thinking 权重与实现没有作为开源版发布。[论文 §4](https://arxiv.org/html/2609.17895v1#S4)

## 值得继续追的问题

1. 固定训练预算、容量和 prior，分别去掉 ECDF、Fourier，再测 ScoringBench 的 CRPS、区间覆盖率和尾部误差；二者是否有交互效应？
2. 3.5 的 grouped prior 对严格 temporal split 是否真有帮助，还是主要由大模型容量、特征工程和 context selection 驱动？
3. 对同一时间任务，分别加入 calendar、合法 lag/rolling、recency context selection，比对随机 / 时间切分和专门序列模型；收益在哪一步出现？
4. Query A 与 Query B 分批、合批、重排时，在固定 train context 和随机种子下预测是否保持一致？若有差异，是数值实现还是外围预处理导致？
5. 在本地开源 3.5、Fast 与 API Plus/Thinking 之间，分别量化质量、延迟、显存及部署/许可代价，避免把 family-best 榜单当成单模型能力。
