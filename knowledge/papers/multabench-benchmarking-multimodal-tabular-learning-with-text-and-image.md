---
type: Research Paper Review
title: "MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image"
description: "MulTaBench 的多模态互补与任务感知表示筛选、TAR 微调及消融分析；结合论文版源码讨论 learner 对表示的利用差异、ICL 收益边界和训练协议非对称性。"
resource: https://arxiv.org/abs/2605.10616
tags: [multimodal-tabular-learning, benchmark, representation-learning, tabular-foundation-model, in-context-learning, lora]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/html/2605.10616v1
    title: "MulTaBench, arXiv v1"
  - id: paper-code
    resource: https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1
    title: "Official paper_version, inspected 2026-10-29"
  - id: current-code
    resource: https://github.com/alanarazi7/MulTaBench/tree/d3864dd18026043e86ef6f6c61624836a09a843b
    title: "Official master snapshot, used only to establish protocol changes"
---

# MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image

## Links

- [arXiv](https://arxiv.org/abs/2605.10616) · [v1 HTML](https://arxiv.org/html/2605.10616v1) · [v1 PDF](https://arxiv.org/pdf/2605.10616v1)
- [官方 GitHub / 项目入口](https://github.com/alanarazi7/MulTaBench) · [论文版本](https://github.com/alanarazi7/MulTaBench/tree/3bb95079a6146ed35d45187e98bcad5b063273a1)
- [当前官方 leaderboard](https://huggingface.co/spaces/multabench/leaderboard) · [当前 Hugging Face 数据入口](https://huggingface.co/multabench)：协议已演进，不能直接当作论文 v1 的实验结果。
- [本次 NotebookLM](https://notebook.google.com/notebook/7ba1ee80-5a93-4b8b-a656-cec8ba59bdde)：收录的是 arXiv 摘要页；讨论另外读取了原文正文及关键附录。

## 一句话结论

MulTaBench 的主要贡献是一个有明确筛选目标的 benchmark：挑出结构化列与文本／图像存在互补收益，而且监督微调 encoder 能改善预测的任务。它提供 40 个数据集，并用多种 learner、encoder 尺度与 embedding 维度验证 Target-Aware Representations（TAR）的收益。

**我们的判断**：值得作为“冻结表示在哪些任务上不够用”的诊断工具。证据支持监督式表示适配的价值，但不足以证明所有 learner 都需要同一种适配方式，也没有充分证明收益必须来自跨模态联合理解。源码更接近“线性头监督的单模态特征学习 → PCA → 表格 learner”；encoder、压缩方案、下游 learner 和训练划分共同决定效果。

## 文章摘要

现有多模态表格 benchmark 常把“多种模态同时出现”当作多模态学习任务的充分条件，但一部分数据集可能被单一模态主导，或通用 embedding 已能表达主要预测信号。作者希望筛出更适合研究任务特定表示的问题。

论文提出两个经验条件：Joint Signal 要求联合输入优于两个单模态基线；Task-awareness 要求微调 encoder 后的联合模型优于使用冻结 embedding 的联合模型。通过 5 个 learner 的共识筛选，最终构成 20 个图像表格任务和 20 个文本表格任务，覆盖分类与回归。随后换 learner、扩大 encoder、改变 PCA 维度并部分移除 PCA，检验收益是否依赖筛选配置。（§3–5）

论文没有提出新的多模态 foundation model。其架构愿景是让未来模型具备任务感知表示能力，同时保留表格 ICL 的低适配成本；这仍是研究方向，而非本文已经实现的结果。（§6–7）

## 方法拆解

### 数据筛选与四组核心对照

| 条件 | 结构化列 | 非结构化模态 | 表示 |
| --- | --- | --- | --- |
| Unimodal Structured | 有 | 无 | 原始结构化特征 |
| Unimodal Unstructured | 无 | 有 | 冻结 embedding |
| Joint Frozen | 有 | 有 | 冻结 embedding |
| Joint TAR | 有 | 有 | 用目标标签微调后的 embedding |

对每个 learner，先对五次运行取平均性能，再计算：

- `Joint gain = Joint Frozen − max(Unimodal Structured, Unimodal Unstructured)`。
- `Awareness gain = Joint TAR − Joint Frozen`。
- 5 个筛选 learner 中，至少 **同样的 3 个**同时满足两项增益 `> 0.001`，数据集才通过；不是两项条件分别找任意 3 个通过即可。

筛选 learner 为 LightGBM、CatBoost、TabM、TabPFNv2、TabPFN-2.5；分类用 AUC，回归用 R²，每个训练 fold 上限 10,000 行。`0.001` 是经验幅度门槛，不等于对每个数据集做了显著性检验。（§3.2、附录 A.2–A.3）

对于图像任务中同时存在的文本列，代码里的 `non` 是“非图像”，可能包含结构化列与文本，不能一律写成纯数值／类别列。三模态补充实验另行拆分文本贡献。

### TAR 的训练目标

E5-small-v2 用于文本，DINOv3-small 用于图像。encoder 最后 3 个 transformer 层插入 LoRA，配线性分类头训练：

`z = encoder(u)` → `logits = Wz + b` → `cross_entropy(logits, target)`。

- 分类直接使用原始类别。
- 回归将连续标签按等频分成 20 桶，遇重复边界时实现可减少桶数；对桶类别做交叉熵，而非 MSE。
- 仅使用当前训练分区，内部再划 train/validation；按 validation cross-entropy 早停并选择 checkpoint。
- 论文设置 LoRA rank 16、alpha 32、dropout 0.1；E5 学习率 `1e-4`、最多 50 epochs，DINO 学习率 `0.001`、最多 100 epochs；batch size 256，patience 3，weight decay 0.01。（附录 A.1）

**encoder 微调不输入结构化特征，也不接收下游 learner 的梯度。**训练完丢弃分类头，导出 backbone embedding，默认每个模态列压缩到 30 个 PCA 分量，再交给表格模型。Joint TAR 替换原来的冻结表示，不是同时追加 Frozen 与 TAR 两套特征。

### 多文本列怎样训练

每个“行－文本列”构成一个样本，输入为 `passage: 列名: 列值`，标签为该行的目标。所有文本列共享同一个 E5，但不会将整行多个文本字段拼接后联合编码。完成微调后，各文本列独立提取 embedding、独立拟合 PCA。

因此，若有 10 个文本列，默认可生成 300 个文本数值特征，而不是所有文本总共 30 维。列名提供字段语义，但其他列的值不进入当前单元格的 encoder 输入。

## 关键实验结果

### 筛选前的结果与筛选偏差

- 作者汇总已有文本 benchmark 的 56 个候选数据集；在筛选前，TAR 已在各筛选 learner 的总体结果上优于 Frozen。
- 约 23% 未通过 Joint Signal；剩余数据集中约 36% 未通过 Task-awareness，最终约 41% 同时通过，再选取 20 个组成文本子集。
- 已有图像 benchmark 的 16 个有效候选中只有 5 个通过，作者继续从公开数据集补充到 20 个。
- 论文数据集规模约 400–114,000 行，结构化特征数 1–245；运行时仍受训练样本上限限制。（§4）

筛选前的结果说明 TAR 收益不是完全由最终筛选制造出来的；但最终 benchmark 明确富集“微调有效”的任务，不能据此估计一般多模态表格任务的获益比例。

### PetFinder 的模态与微调消融

以下为 TabPFN-2.5 的 AUC；S=结构化列，I=图像，T=文本。（表 2）

| 输入／适配条件 | AUC（%） |
| --- | ---: |
| S+I，冻结 | 83.7 |
| S+T，冻结 | 81.0 |
| S+I+T，均冻结 | 84.9 |
| S+I+T，仅图片 TAR | 85.3 |
| S+I+T，仅文本 TAR | 87.3 |
| S+I+T，两者 TAR | 88.0 |

冻结条件下加入第三种模态有收益；固定三种模态后，两种 encoder 的适配也分别有增量。这里“两者 TAR”仍是分别用标签微调两个 encoder，并非图文联合 loss。

### 不同 learner 的 TAR 胜率

胜率定义为同一 learner 在 `(dataset, fold)` 上 `TAR score > Frozen score` 的比例，**不是 learner 之间对打的胜率，也不是平均增益**。（附录 F.1、表 8）

| Learner | 图像胜率（%） | 文本胜率（%） |
| --- | ---: | ---: |
| CatBoost | 90.0 | 93.0 |
| LightGBM | 84.0 | 93.0 |
| TabPFNv2 | 77.0 | 84.4 |
| TabPFN-2.5 | 77.0 | 73.3 |
| TabICLv2 | 55.0 | 75.0 |

TabICLv2 图像胜率的论文 95% CI 为 `55.0 ± 9.8%`，包含 50%。它不支持“图像任务上 TAR 总能稳定帮助 ICL”的强结论；其他 ICL 模型也不能一概归为同样低胜率。

本次讨论曾对论文版自带 CSV 做配对汇总，得到 TabICLv2 图像 100 对结果中 55 次获胜、无平局；按五次运行取数据集均值，20 个数据集为 10 个改善、10 个下降。这只是已发布结果的汇总，不是重新训练或独立复现。下列案例均为五次运行平均 R²：

| 任务／learner | Frozen | TAR |
| --- | ---: | ---: |
| Khaadi Clothes / CatBoost | 0.391700 | 0.648582 |
| Khaadi Clothes / TabICLv2 | 0.812891 | 0.734722 |
| Mango Mass / TabICLv2 | 0.624017 | 0.687184 |

同一任务上，树模型获得大幅收益的同时，TabICL 可能退化；而 TabICL 的 Frozen 绝对性能仍可高于树模型的 TAR。因此“大增益”不能直接等同于“最终最强”，低胜率也不等同于 ICL 本身预测差。结果来源为论文版 `multabench/leaderboard/results/images/` 与 `more_baselines/`，按 `main_paper/leaderboard.py::load_for_overview` 的口径汇总。

### 稳健性与成本

- 新增 XGBoost、RandomForest、RealMLP、TabDPT、TabICLv2 等 learner；总体平均结果仍支持 TAR，但收益稳定性有明显模型差异。
- 换成 Large encoder 后 TAR 仍有收益，作者报告 TAR-small 整体优于 Frozen-large。
- PCA 15／30／60 维下收益仍存在；无 PCA 实验只覆盖 **33 个数据集、CatBoost 和 LightGBM**，不能写成所有 ICL 模型都验证了无 PCA 结论。
- A100 40GB 条件下，文本 small encoder 配 TabPFN-2.5 的单次运行中位数从 242 秒增到 2,396 秒；图像对应配置为 169 秒到 316 秒。均为无 HPO 配置。（§5、附录 F.3–F.5）

## GitHub / Code Analysis

### 版本与分析范围

- 论文版本：`paper_version`，SHA **`3bb95079a6146ed35d45187e98bcad5b063273a1`**。
- 当前版对照：SHA **`d3864dd18026043e86ef6f6c61624836a09a843b`**；其 [protocol 文档](https://github.com/alanarazi7/MulTaBench/blob/d3864dd18026043e86ef6f6c61624836a09a843b/docs/protocol.md#L74-L78) 明确指向论文 tag。
- 当前版已扩展为 80 个数据集、6 folds，并改变多分类／回归指标与缓存实现。本文讨论论文结果时采用论文版，不以当前 leaderboard 回填论文结论。
- 代码采用 MIT license；数据集需分别遵守来源条款。核心实现依赖 PyTorch、Transformers、PEFT、scikit-learn、TabSTAR 工具代码及各 learner 的第三方包；E5／DINO 权重通过 Hugging Face 模型接口加载，不是调用托管预测 API。
- 工作范围为源码拆解；没有训练模型或运行 benchmark。用户明确要求不做复现。该要求提出前仅调用过独立 loss／pooling 函数并汇总自带 CSV，不能视为验证整条训练或评估链路。

### 实现路径与对应事实

以下链接均固定到论文版 SHA。

| 实现 | 已核对事实 |
| --- | --- |
| [benchmark.py](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/benchmark.py#L90-L112) | 模态裁剪与 tune flags 分开；`ft` 按任务类型微调，`ft-img-ft-txt` 同时开启两条独立路径 |
| [TabularModel.fit](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/abstract_model.py#L89-L124) | 先按 learner 决定是否留出 validation，再拟合包括 encoder 在内的预处理 |
| [E5ForTuning / checkpoint](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py#L118-L150) | mean pooling＋线性头＋CE；同文件 280–314 行按 `eval_loss` 选模型，仅返回 backbone |
| [DINOForTuning](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/dino/dino_finetune.py#L97-L125) | CLS token＋线性头＋CE，无结构化输入 |
| [encoder_finetune_loss](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/finetune/utils.py#L17-L24) | 标签展平后直接调用 `torch.nn.functional.cross_entropy`，无回归 MSE 或跨模态辅助 loss |
| [text_embeddings.py](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/text_embeddings.py#L139-L190) | 先按行划分，再展开文本列；共享 E5、逐列 PCA |
| [image_embeddings.py](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/preprocessing/image_embeddings.py#L76-L125) | 图像先微调／编码，再 StandardScaler＋PCA；注释明确提出缓存 embedding 为待改进项 |
| [TabICLv2 wrapper](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/tabicl_v2.py#L7-L22) | 不留出下游 validation，直接对处理后的表格 `fit(x_train, y_train)` |
| [CatBoost wrapper](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/baselines/catboost.py#L11-L39) | 留出下游 validation，2000 iterations、50 轮 early stopping，`use_best_model=True` |
| [胜率生成](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/leaderboard/paper_production.py#L511-L540) | 按 model/dataset/fold 配对，严格 `ft > all`；CI 用配对数的二项正态近似 |

### E5 训练与输出的归一化差异

[E5 训练 forward](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py#L139-L148) 使用 `l2_normalize=False`；[导出 embedding](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/e5/e5_finetune.py#L210-L218) 则调用 `_mean_pool_l2`。源码注释承认一致归一化的效果尚待检验。

[INFERENCE] 训练头可以利用模长，导出时这部分自由度被移除，随后还有 PCA。因此，线性头 CE 的改善不保证完整传递给下游。这是文本路径的潜在目标错配，**不能用来解释图像 TabICLv2 的 55% 胜率**；影响大小没有被本文验证。

论文版文本路径为 L2-normalized embedding 后直接 PCA，图像路径为 StandardScaler 后 PCA，不能笼统描述为所有模态都采用完全相同的标准化流程。

### 不同 learner 未必得到同一套表示

`TabularModel.fit` 在 `fit_preprocessor` **之前**按 `USE_VAL_SPLIT` 划下游验证集。CatBoost／LightGBM 为 True，TabPFN／TabICLv2 为 False。encoder 微调、PCA 都发生在后续预处理里；encoder 又在其收到的训练行内部划一次 validation。

因此，相同外层 split 下，树模型的 encoder／PCA 通常看到扣除下游 validation 后的数据，而 ICL 路径看到完整外层训练分区。论文版在各次 learner 运行内部做 TAR，没有要求所有 learner 读取完全相同的预计算 TAR 矩阵。

**代码结论**：共享“表示学习算法”不等于共享“完全相同的表示及 PCA 坐标”。[INFERENCE] 跨 learner 收益差异混合了表示读出能力与表示拟合数据的差异，不能只归因于架构。单一 learner 内 Frozen／TAR 对照仍然有用。

### 数据边界与消融能力

外层 train/test 在模型 fit 前划分，encoder 和 PCA 位于训练分区内，测试经过 transform 和预测。已读主路径没有显示直接将外层测试标签送入 encoder 微调。内部 validation 后续参与 PCA 或下游训练，不应自动称为测试泄漏。

但随机行划分不等同于患者／实体／时间隔离。当前官方 protocol 也明确承认 group/time 结构未逐数据集审计。这里保留为泛化条件待查，**没有核验具体患者跨 split 重复，不能断言已证实医学数据泄漏**。

论文 CLI 的模态裁剪与 `--tune_e5`／`--tune_dino` 可组合，因而具备构造“单模态＋TAR”的能力。我们指出的缺口是论文核心四组对照没有系统报告它，而不是实现完全不支持。`ft-img-ft-txt` 也没有 cross-attention 或图文联合 loss。

### 胜率与平均图表的口径

胜率缺失任一侧结果的配对会被丢弃，平局不计赢，没有最小增益阈值。CI 按 dataset–fold 数计算，而不是按 dataset 聚类；同一任务不同 fold 的相关性没有在这个公式里建模。

[overview loader](https://github.com/alanarazi7/MulTaBench/blob/3bb95079a6146ed35d45187e98bcad5b063273a1/multabench/leaderboard/main_paper/leaderboard.py#L42-L88) 在加载时将 `test_score` 下截断到 `-0.1`。平均图表进一步筛选模型覆盖完整的数据集，并在 dataset/fold 内做 min-max normalization；胜率表则按各模型可配对结果计算。不能把这两种统计当成同一个命题。

## 关键分析：讨论中需要保留的边界

### 1. 有微调收益，不等于原表示一定丢失信息

用户提出的关键质疑是：不同 learner 对相同多模态特征的利用能力不同。

[INFERENCE] 任务信号可能仍存在于 Frozen embedding，只是分散在下游 learner 难以提取的方向。TAR 可能补充信号，也可能改善几何组织、线性可分性或可读出性。树模型按特征轴切分与 ICL 的归纳偏置不同，但本文未直接量化这些机制。

无 PCA 消融缓解了“收益完全由 30 维压缩制造”的解释，却只覆盖两种树模型，仍不能区分信息恢复与读出改善。比较冻结 encoder＋可学习 adapter／MLP 与 encoder LoRA，才更接近这个问题；本文没有把这种机制对照做完整。

### 2. Joint Signal 不等于必须学习跨模态交互

两种模态各贡献可相加的独立信号，也能通过联合性能筛选。论文的 TAR 还只看单模态与标签，不看结构化条件。

核心对照缺少系统报告 `Unstructured only + TAR`。因此尚未充分回答：encoder 微调后，结构化列是否仍然提供独立增量。如果单模态 TAR 已接近 Joint TAR，就不能把全部收益归为多模态融合。

### 3. ICL 低增益有多种解释，代码不能替代因果实验

[INFERENCE] 一种解释是 ICL 已能充分读出 Frozen 表示，TAR 增量较小；另一种是线性头 CE、归一化／PCA 后的表示与 ICL 不匹配。监督适配也可能改变训练 context 与未见 query 的表示分布，但本文没有验证这种漂移，更不能宣称它只影响 ICL、不影响树模型。

需要同时看 Frozen 绝对水平、TAR 绝对水平、逐任务幅度与胜率。只看“赢了多少次”无法判断哪种解释成立。

### 4. 筛选规则既是贡献，也是结论的边界

作者承认 task property 与 algorithmic solution 被绑定。新 learner 上总体收益能说明筛选不只对某一个模型有效，但并未消除数据选择偏差。

[INFERENCE] 先要求 Joint Frozen 有收益，还可能漏掉“只有让 encoder 结合另一模态条件化后，联合信号才可用”的任务。这类任务反而可能是深层跨模态架构最有价值的地方。因此，MulTaBench 适合作为一类难题集合，不能代表所有多模态学习难点。

## Insights

1. **表示质量应相对于 learner 定义。**可用任务信号、读出能力和有限样本下的可学习性是不同层次。固定一个 learner 的 encoder 排名不必迁移到其他 learner。
2. **Target-awareness 与 row-context-awareness 应分开。**前者知道数据集的预测目标，后者结合当前行其他条件来读文本／图像；本文主要实现前者。
3. **适配可以发生在不同位置。**[INFERENCE] encoder 参数更新、支持集驱动的 token/patch 选择、下游 learner 的 ICL 都可能提供任务适配。需要比较适配位置，而非默认每个任务都做 LoRA。
4. **冻结 encoder 不意味着必须输出一个固定单向量。**[INFERENCE] 可以保留多个 token/patch，再根据带标签支持集选择信息；但若原始压缩已丢失细节，后端无法凭空恢复。
5. **收益应与成本一起评价。**文本 TAR 可增加约一个数量级运行时间；何时值得微调是独立的决策问题，不能由平均正收益替代。
6. **benchmark 的负例同样有价值。**TAR 无收益与退化任务能区分现有表示足够、辅助目标不适合和泛化失败等可能原因；只展示平均增益会掩盖它们。
7. **对比表示利用能力，需要固定表示本身。**统一 outer split 还不够，还应明确 encoder 训练行、内部 validation、PCA 拟合样本和导出变换是否一致。

## 我们的观点

这篇更值得作为 benchmark 设计与经验诊断阅读。它把“模态共存”“模态互补”“任务监督适配”拆开，并提供了一个能暴露 frozen embedding 弱点的数据集合，这部分贡献扎实且实用。

算法实现相对朴素，核心是 LoRA 监督特征提取和下游表格学习。我们认可“任务适配经常有收益”，但对“收益证明通用 embedding 丢失了任务信息”以及“因此需要某种新联合 foundation model”的论证保持保留。尤其是 ICL 与树模型的收益差异、线性辅助目标和跨 learner 预处理划分差异，使机制解释不能只靠总体平均图。

本文最值得带走的问题是：**任务适配应放在 encoder、表示选择还是下游预测阶段，才能兼顾准确性、稳健性与成本？**本文验证了其中一条路线，未证明它对所有 learner 最优。

相关讨论可对照仓库已有 [TabICLv2](tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model.md) 的 structural prior／semantic adaptation，以及 [SwitchTab](switchtab-switched-autoencoders-effective-tabular-learners.md) 中表示作为自动特征候选的观点。

## 值得继续追的问题

- 加入 `Unstructured only + TAR` 后，哪些任务仍需结构化列？收益属于单模态适配还是融合？
- 固定同一套 encoder、PCA 和特征矩阵后，learner 之间的 TAR 收益差异还剩多少？
- 冻结 encoder＋可学习投影／MLP，能否达到 LoRA 的收益？是否真的需要改变原始模态特征提取？
- 回归采用序数／连续目标、文本训练与导出使用一致归一化，会怎样改变不同 learner 的表现？
- 能否通过支持集条件化读取 token／patch，实现无需每个任务梯度更新的任务感知表示？
- 对 TAR 负迁移任务，Frozen 是否已足够强，还是适配目标／checkpoint 选择不匹配？
- 按 dataset 聚类估计不确定性，并审查 patient/entity/time split 后，结论是否仍成立？
- 在同等算力和 HPO 预算下，选择何时微调的策略能否优于无条件 TAR？
