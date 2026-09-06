---
type: Research Paper Review
title: "TabICLv2: A better, faster, scalable, and open tabular foundation model"
description: "TabICLv2 的架构、GraphSCM synthetic prior 与三阶段预训练拆解，并重点讨论 structural prior、domain prior、continued pre-training 和 semantic-aware domain adaptation。"
resource: https://arxiv.org/abs/2602.11139
tags: [tabular-foundation-model, in-context-learning, synthetic-prior, domain-adaptation, tabular-learning]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2602.11139
    title: "TabICLv2: A better, faster, scalable, and open tabular foundation model"
  - id: code
    resource: https://github.com/soda-inria/tabicl
    title: soda-inria/tabicl
---

# TabICLv2: A better, faster, scalable, and open tabular foundation model

## Links

- Paper: [arXiv:2602.11139](https://arxiv.org/abs/2602.11139)
- Official code: [soda-inria/tabicl](https://github.com/soda-inria/tabicl)
- Minimal architecture implementation: [soda-inria/nanotabicl](https://github.com/soda-inria/nanotabicl)
- 本笔记的源码核对基于官方仓库 commit `8f1aa2098c894ab91dba15209cf2002ba4be6c6c`。
- 说明：官方 README 明确写明，当前公开的 TabICLv2 pre-training code 是从原 private pre-training codebase 迁移而来，作者称已经和原代码及 released checkpoints 做过仔细 cross-check，但尚未端到端验证可以完整复现原始预训练结果。因此下面会把“论文/官方说明”和“当前公开代码行为”分开写。

## 一句话结论

TabICLv2 的核心不是单独某个新 Transformer block，而是把三件事一起做强：更适合大表格的分层 Transformer 架构、更大规模的 ICL curriculum，以及比 TabICLv1 / 传统简单 SCM 丰富得多的 synthetic prior。尤其值得保留的是它的 prior 思路：**不是从一种分布采样一张表，而是在随机 DAG 上随机化 source distribution、node transform、parent aggregation、categorical conversion、noise 和可预测性过滤，从而训练模型识别大量不同的“表格生成机制”。**

但这仍然主要是一个非常宽的 **structural/statistical prior**，而不是 semantic/domain prior。它知道“变量之间可能以很多机制发生关系”，却基本不知道某列叫 `age`、`income`、`diagnosis_code` 分别意味着什么。这正好引出我们后面讨论的私有领域适配：continued pre-training、semantic-aware encoder 和 domain-conditioned synthetic prior 是三个互补方向。

## 文章摘要

TabICLv2 延续 TabICL 的 tabular in-context learning 路线：给模型一张表里的 `(X_train, y_train, X_test)`，模型在一次 forward pass 中从训练样本推断当前数据集的预测规律，再输出 test prediction，不需要为每个下游数据集重新训练一套模型。

官方实现支持 classification 和 regression。模型的学习能力来自在大量 synthetic datasets 上做 meta-pretraining；推理时再通过 feature / class permutations、normalization 等形成 ensemble views。公开 README 将 v2 的提升归因于三类因素：更好的 synthetic pre-training data、architecture improvements 和更好的 pre-training recipe。

架构上仍然是三层次处理：

```text
columns / feature values
    ↓
column-wise Transformer
    ↓
每个 feature 的 contextual embedding
    ↓
row-wise Transformer + CLS aggregation
    ↓
每一行的 representation
    ↓
dataset-wise ICL Transformer over rows
    ↓
classification / quantile-regression prediction
```

这使它把“列内/列间结构”和“样本间 ICL”拆开处理，而不是对整张 `n × d` 表直接做一个二维全连接 attention。

## 模型架构拆解

### 1. Column-wise embedding

当前官方 `TabICL` 实现中的 `ColEmbedding` 使用 induced self-attention / Set Transformer 风格的 bottleneck attention。v2 classifier 预训练脚本的配置是：

- `embed_dim=128`；
- 3 个 column blocks；
- 8 heads；
- 128 inducing points；
- feature group size 3；
- target-aware column embedding；
- column attention 启用 query-aware scalable softmax，公开脚本中为 `qassmax-mlp-elementwise`。

这里的目标是：先把每一列放到当前 dataset 的上下文中编码，而不是假设“第 3 列”在所有数据集里具有固定语义。

### 2. Row-wise interaction

Column representation 再进入 row-wise Transformer。公开 v2 classifier recipe 使用 3 blocks、8 heads 和 4 个 learnable CLS tokens。多个 CLS representation 最后拼接，形成给 ICL 模块使用的 row representation。

源码也保留了 v1/v2 的 RoPE 差异：当前 v2 预训练脚本设置 `row_rope_interleaved=False`，并使用 standard residual initialization（`zero_init=False`）；这些属于公开实现里明确存在的 v2 recipe 差异。

### 3. Dataset-wise in-context learning

最后的 `ICLearning` 在 row representations 上做 dataset-level Transformer。公开 classifier recipe 使用 12 blocks、8 heads，并同样开启 `qassmax-mlp-elementwise` scalable softmax。

分类时模型把训练行的 `y_train` 编进上下文并预测 query rows；回归则输出 quantiles。当前代码还实现了 KV caching，用于同一训练集上反复预测时复用 training context 的 key/value projection。

## TabICLv2 synthetic prior：到底怎样生成一张表

这是我们讨论最深入的部分。最准确的心智模型不是：

> 从某个分布采一批叶子节点，再反复做 sum / product 得到 target。

而是：

```text
sample dataset shape / feature types
        ↓
sample a random DAG
        ↓
assign X / y features to graph nodes
        ↓
evaluate only ancestors that can affect outputs
        ↓
source nodes: sample RandomPoints → random transform
non-source nodes: transform parent states → combine / concatenate
        ↓
node-level normalization / importance / noise
        ↓
categorical or numerical converter
        ↓
assemble X / y
        ↓
dataset preprocessing + predictability filtering
```

也就是说，prior 的随机性存在于很多层，而不是只在初始分布。

### 1. DAG 怎么生成

当前 `RandomDAG` 实际由 `RandomCauchyDAG` 生成。源码中会给节点采样 heavy-tailed / Cauchy 风格的 input/output importance，再加入全局 offset，通过 sigmoid(logits) 得到边出现概率；边只允许由较小 node index 指向较大 node index，因此天然不会出现 cycle。

此外，`RandomDataset` 在 `filter_unpredictable_graphs=True` 时会检查 X 与 y 是否共享祖先。如果生成的图让输入与目标完全无共同生成因素，会丢掉重采。这一点很重要：**prior 不是盲目接受任意 DAG，而是在图层先排除明显不可学的任务。**

`RandomGraphFunction` 还会剪掉不影响输出 feature 的节点，所以真正计算的是与 X/y 输出相关的祖先子图。

### 2. “root node” 的术语容易误导

我们前面纠正过一次这里的方向。

TabICLv2 代码里说的 `root node`，实际上是 **没有 parent 的 source node / indegree-0 node**。如果按 causal generation 的箭头画：

```text
A  →  B  →  C  →  D
```

那么代码会把 A 叫 root/source，因为值从 A 开始生成；但如果把最终输出 D 画在一棵 dependency tree 的最上方，人类视觉上又会把 D 叫“根”，A 看起来像“叶子”。

所以以后最好不用“树的根/叶子”来记，直接说：

- `source node`：无父节点，自己采样 latent values；
- `derived node`：有父节点，由父节点经过随机函数生成。

这样不会再被画图方向干扰。

### 3. source node 并不是都从同一种分布取样

`RandomPoints` 会在多类基础几何/概率结构之间随机选择。当前 active choice 明确包括 uniform、Gaussian、`RandomCirclePoints`（更准确地说是 unit-ball-like points）以及对 uniform / Gaussian 再施加随机 covariance transform 的 points；采出来之后还会再经过一个 `RandomFunction` 做额外随机变换。

因此“所有叶子都从标准高斯取样”这个理解不对。更准确的是：**source distribution 本身就是 prior 的一部分，而且后面还会被随机 nonlinear mechanism 再扭曲。**

### 4. derived node 也不是固定 `sum/product/max/logsumexp`

非 source 节点使用 `RandomMultiFunction` 处理 parents。它至少包含两种高层路径：

1. 把多个 parent representation concat，然后整体送进一个随机函数；
2. 每个 parent 先各自经过随机函数，再通过一个 aggregation op 合并。

第二条路径里的 aggregation op 才会从下面四种中采：

```text
sum
product
max
logsumexp
```

所以这四个操作不是“每个非叶节点唯一能做的 transform”，只是 multi-parent composition 的一类机制。

### 5. `RandomFunction` 的函数族非常宽

当前公开 prior 里至少能看到这些 function family：

- MLP；
- random tree；
- random discretization；
- linear；
- quadratic；
- generalized GP-like function；
- EM-inspired soft assignment；
- 两个 cheap random functions 的 product。

其中 generalized GP 实现使用 random-feature 风格的频率/矩阵变换，并可随机启用 product kernel；random tree 是随机 split / leaf value 的 symmetric/oblivious tree ensemble；quadratic 为控制开销最多抽取 20 个输入维度。

这解释了为什么把 TabICLv2 prior 简化成“SCM + MLP”是不够的：它真正想覆盖的是一大片不同的局部 functional mechanisms。

### 6. node 级别还有额外随机化

`RandomNodeFunction` 还会随机 node latent dimension，并在 mechanism 前后叠加诸如：

- standardization / normalization；
- L2 normalization；
- random feature importance；
- Gaussian noise；
- node importance；
- numerical / categorical converters。

分类 feature 也不是简单把连续数值 round 成整数；converter 里有 neighborhood/discretization/softmax sampling 等不同 categorical construction path。

因此一张 synthetic table 的统计形态来自很多随机机制叠加，而不是一个“真值 SCM”直接吐出所有列。

### 7. dataset 级别会过滤“完全学不到”的任务；trivial filter 是可选项

当前 prior pipeline 还有 dataset-level predictability check。公开实现使用浅层 `ExtraTreesRegressor` 的 out-of-bag prediction 与 mean baseline 比较，并通过 bootstrap 检查模型是否显著优于无信息 baseline；classification target 会先转成 one-hot representation。

`filter_unpredictable_datasets=True` 时，不够可预测的数据集会被过滤。代码还提供 `remove_trivial_datasets` 路径，用来去掉几乎可以被浅层树解决的过度简单任务；但当前三个 v2 classifier reference pretraining scripts **没有启用这个开关**，其默认值也是 `False`。

因此对当前 v2 reference training，更准确的理解是：

> 先大范围随机生成，再用 graph-level ancestor overlap 和 dataset-level predictability check 保证任务至少存在可学信号；代码具备进一步过滤 trivial tasks 的能力，但 released reference recipe 没有启用这一层。

这比单纯增加 generator complexity 更重要，因为 foundation model 真正学到的是 **task distribution**。

## GraphSCM 输出到真正训练 tensor 还做了什么

`GraphSCM` 生成混合 numerical/categorical features 后，还会经过训练入口的统一处理：

- X 做 outlier removal；
- standard scaling；
- 可做 feature permutation；
- feature 数不足 `max_features` 时 zero pad；
- regression y 同样做 outlier removal / scaling；
- classification y 可做 class-label permutation；
- NaN / 异常样本会走保护逻辑，避免污染整个 batch。

所以“prior”实际上跨越 graph generator、feature converter 和 training preprocessing，不能只看 `graph.py`。

## 三阶段预训练 curriculum

公开 classifier scripts 对 v2 的 curriculum 写得非常清楚：

| Stage | Steps | 每个 synthetic dataset 的 samples | Train fraction | Max LR | Grad clip | 主要目的 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 1 | 500,000 | 1,024 | 0.30–0.90 | `8e-4` | 10 | 大规模学习 broad prior / basic ICL |
| 2 | 40,000 | 400–10,240，log-uniform | 0.79–0.81 | `1e-4` | 10 | 扩到中大型 context |
| 3 | 10,000 | 400–60,000，log-uniform | 0.79–0.81 | `2e-5` | 1 | 适配超长 dataset/context |

三阶段共同使用：

- batch size 64；
- Muon optimizer；
- `graph_scm` prior；
- 最多 100 features；
- `weight_decay=0.01`；
- column / ICL attention 都启用 query-aware scalable softmax；
- stage 2/3 启用 FlashAttention-3；
- synthetic datasets 在 DataLoader workers 中 **on the fly** 生成。

Stage 3 最多 60K total samples，而 train fraction 约 80%，所以官方 README 说的“pre-trained on up to about 48K training samples”和脚本里的 `max_seq_len=60000` 并不矛盾。

### paper-vs-code：cautious weight decay 的复现差异

这是一个值得明确记录的实现事实。

论文文字报告使用 cautious weight decay；当前代码的 Muon 也实现了对应分支。但官方 v2 stage scripts 明确把 `--use_cautious_wd False`，并在注释中解释：reference pretraining runs 里这个开关没有真正接到 Muon，因此 released checkpoints 实际没有使用 cautious WD。公开脚本选择复现 checkpoint 的真实训练行为，而不是机械照论文文字打开它。

这类差异以后复现 foundation model 时应该优先看：**released checkpoint 对应的真实训练 path > 论文配置表的单行描述。**

## 关键实验与能力边界

当前本地一手材料能够稳定核对的官方结论包括：

- 官方 README 将 TabICLv2 描述为 TabArena 和 TALENT 上的 state-of-the-art classification/regression model；
- 在 TabArena 上，不做 per-dataset hyperparameter tuning 的 TabICLv2，官方称在约 80% datasets 上超过经过重度 tuning 的 XGBoost / CatBoost / LightGBM；
- H100 上，50,000 samples × 100 features 的 `fit + predict` 官方报告小于 10 秒，并称约比 TabPFN-2.5 快 10×；
- 预训练覆盖大约 300 到 48K training samples、2 到 100 columns；公开实现/benchmark 显示还能外推到更大的 row/feature counts，但这属于超出 pretraining support 的 generalization，而不是 prior 本身已经覆盖到那些范围。

这里暂不把图中的 average rank 或论文表格数字抄成精确数值，因为当前 NotebookLM source 在本地 sandbox 中无法联网重新取回原文，且我们不希望凭二手记忆补数字。后续如果重新拿到 PDF/NotebookLM source，可以再把完整 benchmark table 补进来。

## 我们围绕 domain prior 的讨论

### 先区分：TabICLv2 的 prior 是 structural prior，不等于 domain prior

TabICLv2 的 GraphSCM 很强，因为它覆盖很多：

- graph topology；
- functional mechanisms；
- source distributions；
- interaction order；
- categorical construction；
- noise / scaling / importance；
- dataset difficulty。

但它几乎不显式建模：

- feature name / description；
- unit；
- schema role；
- domain ontology；
- 某个业务领域真实允许/不允许的 causal mechanism；
- 领域特有 missingness / censoring / selection mechanism。

所以“随机函数足够丰富”仍不等于“懂医学/金融/工业等领域”。前者覆盖统计结构，后者需要条件化的语义和机制先验。

### 私有领域可以自己合成数据做 training 吗？

可以，而且这和 TabICLv2 的训练范式非常兼容。真正关键不是追求 synthetic row 看起来和真实 row 一模一样，而是让 synthetic **task distribution** 更接近私有领域会遇到的预测问题。

可行的 domain prior 可以从下面几层逐步增加约束：

```text
generic GraphSCM
    ↓
匹配 domain 的 feature count / categorical cardinality / missingness / noise
    ↓
匹配 domain 中常见 dependency graph 与 mechanism family
    ↓
根据 schema / metadata / ontology 条件化 generator
    ↓
generic prior + domain prior 混合 continued pre-training
```

不建议一开始完全替换 generic prior。保留一部分 broad synthetic tasks 可以减少模型过拟合到狭窄领域生成器，或把 domain generator 自身的偏差学成“世界规律”。

## 私有领域适配的三个方向

### 方向 1：continued pre-training / domain episodic training

这是最直接的一条。

ICL foundation model 的训练单位本来就是“一个 dataset / task episode”，因此多个私有小表并不需要先做全局 row alignment，也通常不需要把所有 feature 对齐成同一 schema。可以让：

```text
dataset A → episode A
dataset B → episode B
dataset C → episode C
...
```

然后在 meta-batch 中把这些 episodes 混起来 continued pre-train / fine-tune。

所以我们前面说的“小数据加和成大数据”准确含义是：**把很多小数据集累积成更大的 task distribution**，不是把它们强行 row-wise concat 成同一张大表。

仍然需要对每个 dataset 内部做一致的 preprocessing、target definition 和 context/query split，但不要求 `dataset A.column_3 == dataset B.column_3`。

这正是 tabular ICL 对私有数据特别有吸引力的地方：跨 dataset 的统一接口比传统单模型 supervised training 更自然。

### 方向 2：semantic-aware column encoder

这里的 “semantic-aware” 是让模型不只看到匿名数值列，还能看到列的 metadata，例如：

```text
feature name
description
dtype
unit
allowed values / category descriptions
schema/table role
ontology concept
```

例如匿名模型只能看到：

```text
col_1 = 67
col_2 = 1.72
```

semantic-aware 模型还知道它们分别是 `age_years` 和 `height_m`。这样跨数据集时，即使 column position 完全不同，也可以通过语义把相近 feature 放到相近 representation。

我们讨论时的判断是：**第一版通常没必要重训整个 backbone，主要新增/训练一个 semantic encoder 或 adapter 就够形成可验证实验。** 典型做法是把 metadata encoder 输出与原来的 value/statistical column embedding 融合：

```text
value encoder ─────────┐
                      ├─ fused column representation → TabICL backbone
metadata/text encoder ─┘
```

可以先冻结大部分 TabICL，只训练 metadata encoder + fusion layer；如果有足够 domain episodes，再逐步解冻 column encoder / backbone。

### 方向 3：domain-conditioned synthetic prior

第三条是直接改 TabICLv2 最核心的 synthetic task generator，让 prior 对 domain 有条件。

例如不再全局随机采任意 DAG，而是：

- 根据 schema role 限制可出现的 edges；
- 对不同 feature type 采不同 source distributions；
- 对特定 node pair 选择更合理的 mechanism family；
- 学习真实 domain 的 missingness、cardinality、noise、outlier 和 censoring；
- 用 feature metadata / ontology 作为 GraphSCM 的 conditioning signal。

这条路线最“foundation-model-like”，因为它不只是在真实数据上继续训，而是在改变模型未来会遇到的 **hypothesis space / task prior**。

## 三个方向怎么组合

三者并不是替代关系：

```text
generic GraphSCM prior ───────────────┐
                                     ├─ mixed episodic continued PT → domain TabICL
domain-conditioned synthetic prior ──┤
                                     │
real private datasets ────────────────┘

feature/schema metadata → semantic encoder → column representation
```

比较务实的实验顺序是：

1. **先做 continued PT**：验证私有 domain 的多小表能否作为 episodes 带来稳定收益；
2. **再加 semantic encoder**：它改动小，且能直接验证“跨表 semantic alignment”是否值得；
3. **最后做 domain prior**：如果前两者证明 domain signal 有价值，再把 signal 前移到 synthetic generator，做更系统的 domain-conditioned pretraining。

这比一开始就训练一个复杂生成模型风险更低，也更容易定位收益来自哪里。

## 关键分析

### 1. prior 的价值在覆盖 mechanism，而不只是覆盖 marginal distribution

如果只匹配每列的均值、方差、类别占比，再采独立 synthetic rows，模型学不到真正的预测结构。TabICLv2 的设计说明更值得模拟的是：

> 哪些变量相关、如何组合、关系多非线性、噪声多大、任务有多难。

这对私有 domain synthetic data 同样成立。

### 2. domain prior 不一定要求生成“可读的真实记录”

如果目标是 foundation-model pretraining，synthetic data 的主要功能是提供 learning problems，而不是通过人类肉眼的 realism test。

因此 domain generator 可以是半抽象的：保留 domain 的 dependency pattern、cardinality、missingness 和 mechanism family，但不必还原真实个人/实体。这在隐私领域尤其有吸引力。

### 3. semantic-aware 和 domain prior 解决的是不同问题

- semantic encoder：告诉模型“这些列是什么”；
- domain prior：告诉模型“这些变量通常怎样生成、怎样相互作用”；
- continued PT：告诉模型“这个领域真实任务的 empirical distribution 是什么”。

把三者区分开后，实验设计会清楚很多。

## Insights

### Insight 1：Tabular FM 的可迁移单元可以是 dataset，而不是 row

传统 supervised learning 常把“更多数据”理解成更多 rows；ICL meta-training 允许把多个 schema 不同的小数据集累积成更多 tasks。这个差别对企业私有数据尤其重要，因为真实组织里常见的恰恰是“很多小表”，而不是“一张统一超大表”。

### Insight 2：synthetic prior 的设计空间比 model architecture 更值得系统研究

TabICLv2 已经说明，一个很强的 prior 可以通过大量随机机制组成，而不是试图找到一个唯一正确的 SCM。对于 domain adaptation，值得问的问题也应从“生成器像不像真实数据”转成“训练 task distribution 是否覆盖真实领域的 decision problems”。

### Insight 3：semantic metadata 是跨 schema 对齐的一种自然接口

如果模型完全匿名化 feature，那么跨 dataset 只能靠数值分布和任务上下文自己推断 feature role；加入 names/types/units/schema 后，可以显式给模型一条跨表对齐通道。这可能比手工做全量 dataset schema alignment 更轻。

### Insight 4：generic prior 与 domain prior 最好混合，而不是二选一

纯 generic prior 缺 domain bias；纯 domain prior 又可能过拟合 generator。一个更稳的方案是 mixture pretraining，让模型同时保留 broad mechanism coverage 和目标领域偏置。

## 我们的观点

TabICLv2 对我们最有启发的不是“又一个 TabPFN competitor”，而是它把 **synthetic task distribution 当成模型能力的一等公民**。GraphSCM 里大量随机化看起来很工程，但它回答了一个很实用的问题：如果没有足够真实 supervised datasets，怎么制造足够多、足够不同、又至少可学的 tabular tasks 去训练 ICL learner。

它的明显空白也正好是潜在研究空间：现有 prior 很宽，但很匿名。对私有垂直领域而言，我们更关心的下一步不是继续无条件扩大 random function zoo，而是：

> 能不能用真实领域的小数据、schema metadata 和业务机制约束，把 generic structural prior 变成 domain-conditioned prior，同时保留 ICL 的跨数据集适配能力？

从工程优先级上，continued PT + 一个轻量 semantic encoder 是最低成本验证；如果这两者已经有 signal，再投入 domain-conditioned GraphSCM 会更合理。

## GitHub / Code Analysis

### 公开实现的关键路径

本次实际核对的核心文件位于官方 repo：

```text
src/tabicl/_model/tabicl.py
src/tabicl/_model/layers.py
src/tabicl/_model/ssmax.py
src/tabicl/prior/_graph_scm.py
src/tabicl/prior/graph_lib/README.md
src/tabicl/prior/graph_lib/_dataset.py
src/tabicl/prior/graph_lib/_graph.py
src/tabicl/prior/graph_lib/_node_function.py
src/tabicl/prior/graph_lib/_points.py
src/tabicl/prior/graph_lib/_multi_function.py
src/tabicl/prior/graph_lib/_function.py
scripts/train_v2_clf_stage1.sh
scripts/train_v2_clf_stage2.sh
scripts/train_v2_clf_stage3.sh
```

从调用关系看，prior 是层级 sampler，而不是单个 generator class：

```text
RandomDataset
  → RandomDAG
  → RandomGraphFunction
      → RandomNodeFunction
          → RandomPoints                 # source nodes
          → RandomMultiFunction          # derived nodes
              → RandomFunction
                  → random matrix / activation / weights / tree / GP / ...
```

### 代码事实与论文描述的两个复现注意点

1. v2 pretraining code 是迁移版，官方尚未声称做完 end-to-end checkpoint reproduction；
2. cautious weight decay 是明确的 paper-vs-reference-run discrepancy，released checkpoints 对应 `use_cautious_wd=False`。

因此未来如果做严格 reproduction，不能只拿论文超参数表重新实现一遍；应以当前官方 scripts + checkpoint config 为基准，并把上述差异写进实验记录。

## 值得继续追的问题

1. GraphSCM 中各 function family 的 ablation 到底各贡献多少？是“函数越多越好”，还是少数 mechanism family 占主要收益？
2. predictability filter 对最终 ICL 能力的影响有多大？过强过滤是否会让模型对 noisy / weak-signal real datasets 不够鲁棒？
3. domain continued PT 中，generic synthetic / domain synthetic / real private episodes 的最佳 mixture 比例是多少？
4. semantic encoder 只用 feature name 就够不够？加入 description / unit / ontology 后收益是否稳定？
5. domain-conditioned prior 应该手工定义机制族，还是从真实 domain tables 学一个 hyper-prior 来控制 DAG 和 functions？
6. 多个私有 dataset 的 continued PT 是否会出现 domain-level catastrophic forgetting，以及 generic prior replay 能否解决？
7. classifier 和 regressor 的 prior / architecture 配置差异中，哪些是任务本质所需，哪些只是当前 recipe 的工程选择？
