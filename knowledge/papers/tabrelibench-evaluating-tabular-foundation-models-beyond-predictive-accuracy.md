---
type: Research Paper Review
title: "TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy"
description: "TabReliBench 的多维可靠性评估、模型排名与 WSC 实现；结合价差预测讨论局部短板、ScoringBench 指标和分位数校准的适用边界。"
resource: https://arxiv.org/abs/2610.13654
tags: [tabular-foundation-model, benchmark, probabilistic-regression, calibration, quantile-regression, subgroup-reliability]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2610.13654v1
    title: "TabReliBench, October 2026, v1"
  - id: code
    resource: https://github.com/Tianqi-Zhao/TabReliBench/tree/f538f3b5b230495b1fcb2dadf528a123156a3d99
    title: "Official implementation, inspected 2026-10-16"
  - id: scoringbench
    resource: https://arxiv.org/abs/2603.29928
    title: "ScoringBench: A Benchmark for Evaluating Tabular Foundation Models with Proper Scoring Rules"
---

# TabReliBench: Evaluating Tabular Foundation Models Beyond Predictive Accuracy

## Links

- [论文 arXiv:2610.13654](https://arxiv.org/abs/2610.13654) · [v1 PDF](https://arxiv.org/pdf/2610.13654v1) · [v1 HTML](https://arxiv.org/html/2610.13654v1)
- [官方 GitHub](https://github.com/Tianqi-Zhao/TabReliBench)；论文与代码入口均指向该仓库，未确认另有独立官方项目页。
- [本次讨论的 NotebookLM](https://notebook.google.com/notebook/268caf73-1b92-4d77-a3bb-a3b796396e1d?authuser=1)：通过 source fulltext 读取其论文来源，并核对 canonical arXiv 链接。
- 对照：[ScoringBench 论文](https://arxiv.org/abs/2603.29928) · [官方代码](https://github.com/jonaslandsgesell/ScoringBench)；仓库已有 [TabPFN-3.5 笔记](tabpfn-3-5-technical-report.md)中的 ScoringBench 讨论和 [Causilo 笔记](causilo-technical-report.md)。这里仅作对照，不新增 ScoringBench 独立条目。

## 一句话结论

这篇的主要贡献是 benchmark 和 empirical diagnosis：把 TFM 的点预测、分布质量、边际校准、子群覆盖与区间效率拆开评估。TFM 原生回归分布整体表现较强，但模型排名随数据集和指标变化；好的 R² / CRPS 或整体 coverage 不能代替局部可靠性检查。Ensemble 与 conformal 也没有稳定修复所有校准问题。（§2–6）

**我们的判断**：最值得保留的是模型对比表和评估方法。它与用户观察到的“TFM 整体强、少数市场阶段明显输给 LightGBM”相呼应，但没有直接验证时间 regime 下的相对优势反转，也没有提出补齐 TFM 局部短板的有效训练方法。

## 文章摘要

很多 TFM 输出完整 predictive distribution，但常见 leaderboard 主要评价点预测。作者构建 TabReliBench，在 160 个分类和 81 个回归数据集上比较 TFM 与传统方法，并检查 accuracy / R²、Brier / CRPS、ECE / PIT、整体和子群 coverage 以及区间宽度。（§1–2）

分析进一步区分 dataset、model 和 dataset–model interaction，使用训练数据 meta-features 预测平均表现与模型相对优势，最后进行 ensemble size 和九种 conformal 后处理实验。主要结果是：预测分数的变化很大程度由数据集决定；相对 TFM 优势、尤其 coverage 的变化更难预测；更好的整体分布评分不保证更好的直接校准诊断。（§3–6）

## 方法拆解：这篇到底做了什么

### 数据、模型与评估单位

- 数据来自 OpenML；每个数据集生成 5 次随机划分，每次无放回抽取至多 10,000 行，约一半训练、一半测试。分类做分层抽样，各模型使用相同观测与划分。它不是 temporal / walk-forward benchmark。（§2；附录 B）
- 11 个分类 TFM：TabPFNv2 / v2.5 / v3 / v3.5、TabICLv1 / v2、TabDPT1.1 / 1.3、Causilo、LimiX-2、TabFM。
- 9 个回归点预测 TFM；其中 LimiX-2 和 TabFM 在本文配置中不提供回归分布，所以分布比较只有 7 个 TFM。不能据其 R² 排名判断 quantile 能力。
- 传统配置为 BART、RealMLP / RealMLP-HPO、XGBoost / XGBoost-HPO；**主榜没有 LightGBM**。主 TFM panel 用 8 个 ensemble members。（§2；官方 README）
- 通常先对每个 dataset–model 的 5 个 split 求均值，再做数据集内排名，最后等权平均数据集排名。跨数据集样本量是数据集数，重复 split 不能增加这个统计单位的数量。

### 为什么把指标拆开

| 维度 | 分类 | 回归 | 含义 |
| --- | --- | --- | --- |
| 点预测 | Accuracy | R² | 点预测是否准确 |
| 分布质量 | Brier | CRPS | Proper score 综合评价概率预测；不能解释为纯校准分数 |
| 边际校准 | Confidence ECE | PIT–KS、marginal coverage | 概率/区间整体是否匹配观测频率 |
| 类别 / 子群 | Classwise ECE | Worst-slab coverage（WSC） | 平均值是否掩盖局部失准 |
| 区间效率 | — | Normalized interval width | 在相近 coverage 下是否足够窄 |

分类 ECE 使用 15 个近似等样本量 bins。回归统一为 199 个 quantile levels，0.005–0.995，步长 0.005；CRPS 用 Pinball loss 的梯形积分近似，不加端点项。归一化 CRPS 用该 split 的训练目标标准差；区间宽度的标准化用 evaluation 目标标准差，两者不是同一分母。（Table 1；附录 C）

### WSC 是什么，具体评估什么

Subgroup 是按输入特征划出的一部分样本。整体 90% coverage 可以由“80% 样本 coverage 为 95%、另外 20% 为 70%”组成；后者是整体平均掩盖的欠覆盖。这是说明性例子，不是论文实验数字。

本文 WSC 的具体过程：（附录 C.3；实际代码见后文）

1. 把已随机排序的测试数据分成 search / evaluation 两半。
2. 每次只检查一个特征。数值特征初始按五个分位 bins 划分，小组与相邻组归并；类别特征从类别组开始归并小组。
3. 每个候选组至少包含 `max(1, floor(n_search / 5))` 个 search 样本；缺失值不属于候选组，高基数类别特征会跳过。
4. 在 search 半中选 coverage 最低的组，把相同边界或类别组合应用到 evaluation 半，再计算该组 coverage。
5. 没有有效组或 evaluation 组为空时，WSC 未定义，不能填成 0 或 1。

**边界**：它测的是搜索到的单特征 slab 的 held-out coverage，不是所有特征组合的最差 coverage，也不是每个 x 的 conditional coverage 保证。约 20% 的 search support 门槛也使其不适合直接检测极小的稀有事件子群。每个模型可能选择不同的 slab，所以 WSC 不能替代同一业务子群上的模型配对比较。独立 evaluation 中不保证 WSC 一定低于 marginal coverage。

论文把 `|MC − nominal| + |WSC − nominal|` 作为 total coverage error；绝对值先在 split 内计算，再聚合。WSC 对应的图中 width 仍是整体 evaluation width，并不是该 slab 内的 width。（附录 C）

### 分解变化与预测可靠性

作者对 5-split 平均分数做 dataset + model + interaction 的平方和分解。预测指标在 TFMs 之间更多呈现共同的数据集难度变化；coverage 的 interaction 和 split 相关波动更明显。加入传统模型后，model / interaction 占比也会变化，所以不能推广成“任何模型差异都不重要”。（§3.3；附录 D.4）

训练数据上提取 62 个分类、59 个回归 meta-features，按 X、Y、XY、Aux 累加，用 ridge 和 Extra-Trees 在跨数据集 5×5-fold CV 中预测共同分数变化与模型相对偏离。共同预测分数较容易预测，相对 TFM 优势大多难预测，PIT / coverage 的共同变化也较难预测。（§4.1）

## 关键实验结果：哪些模型比较好

### 平均排名：越小越好

下表摘自附录 Tables 6、7、9。R² 有 14 个配置，CRPS / PIT / coverage 有 12 个配置；不同列不能横向比较名次数值。Coverage error 使用 90% nominal coverage。

| 回归模型 | R² rank | CRPS rank | PIT–KS rank | MC error rank | WSC error rank | Total error rank |
| --- | --- | --- | --- | --- | --- | --- |
| LimiX-2 | **3.30** | — | — | — | — | — |
| TabPFNv3.5 | 4.05 | **2.33** | 4.37 | 4.31 | 4.34 | 4.10 |
| TabFM | 4.42 | — | — | — | — | — |
| Causilo | 4.54 | 3.21 | **3.32** | **3.86** | **4.02** | **3.80** |
| TabPFNv3 | 5.02 | 3.46 | 4.54 | 5.41 | 5.71 | 5.44 |
| TabDPT1.3 | 5.75 | 4.77 | 5.49 | 5.09 | 4.77 | 4.84 |
| TabPFNv2.5 | 6.12 | 4.16 | 4.36 | 5.49 | 5.31 | 5.33 |
| TabPFNv2 | 7.17 | 5.30 | 4.79 | 5.12 | 4.85 | 4.80 |
| TabICLv2 | 7.25 | 5.65 | 3.49 | 4.31 | 4.28 | 4.01 |
| BART | 10.54 | 9.79 | 7.95 | 6.44 | 6.74 | 7.26 |
| RealMLP-HPO | 10.04 | 8.28 | 8.40 | 8.27 | 8.16 | 8.36 |
| XGBoost-HPO | 12.23 | 10.14 | 10.64 | 10.49 | 10.53 | 10.64 |

分类平均排名：accuracy 领先的是 LimiX-2 **4.66**、TabFM **4.77**、TabPFNv3.5 **4.91**；Brier 领先的是 TabPFNv3.5 **3.77**、TabFM **3.82**。Confidence ECE 最好是 Causilo **5.11**，classwise ECE 最好是 TabPFNv3.5 **5.21**。这些是特定配置的平均排名，不能把小名次差直接写成显著胜出。（Table 6）

### 绝对分数与排名不是同一聚合

Table 13 报告各数据集 5-split 均值的跨数据集中位数：

| 回归模型 | R² ↑ | Normalized CRPS ↓ | PIT–KS ↓ | Total coverage error ↓ |
| --- | --- | --- | --- | --- |
| TabPFNv3.5 | 0.881 | 0.134 | 0.0473 | 0.0559 |
| Causilo | 0.885 | 0.143 | 0.0458 | 0.0502 |
| TabICLv2 | 0.871 | 0.145 | 0.0471 | 0.0541 |
| BART | 0.846 | 0.195 | 0.0887 | 0.0983 |

Causilo 的 PIT 平均排名最好，不代表其跨数据集中位 PIT 数值也最小。平均排名、分数中位数以及 paired difference 回答不同问题。Total error 是两个绝对偏差之和，不能解释为“整体漏掉了这么多样本”。（附录 C.4、D）

### Ensemble、conformal 与 meta-features

- E 从 1 到 32：11/11 分类 TFM 的 median accuracy 和 Brier 改善，7/7 分布回归 TFM 的 normalized CRPS 改善；median R² 只有 6/9 改善，TabPFNv3.5 / v3 / v2.5 的中位数反而下降。多数收益发生在小到中等 E。（§5.1；附录 H）
- 同样的 E 变化下，classwise ECE 仅 5/11 改善，PIT–KS 仅 3/7 改善。更大的 ensemble 不是稳定的局部校准修复器。
- 九种 conformal 后处理应用于冻结的 E=8 分布预测。SCP-CQR 对原本明显 overcover 的 BART 收窄区间、改善 marginal coverage；TFM 本来接近 nominal，收益较小。没有一种后处理跨模型、跨 nominal level 稳定消除 WSC gap；不能据此说 conformal 没有边际覆盖价值或理论保证。（§5.2）
- 局部方法还可能退化成无限区间。例如 PCP 的跨 base-model 平均 infinite-interval rate 随目标 coverage 从 80% 到 95% 由 **14.3%** 升到 **44.1%**；coverage 增益必须与信息量一起解释。（附录 G）
- 分类 neighborhood disagreement 与 accuracy 的 Spearman ρ 为 **−0.78**，与 confidence ECE 为 **0.64**。回归训练样本量 n 与 WSC error 的 ρ 为 **−0.73**。这些是跨数据集关联，不能推出市场高波动等逐时刻变量已经被证明能预测失准。（§4.2，Figures 5–6）

## 关键分析：证据的边界

1. **主要是 empirical / benchmark 贡献。** 指标本身多为已有方法；价值是统一比较、分开解释、检查 ensemble / conformal 的实际收益。
2. **回归 baseline 配置影响结论。** RealMLP 和 XGBoost 训练的是 199 个 quantile 的平均 Pinball objective；点预测由 quantile function 积分产生，不是独立针对 MSE 优化的常规回归配置。HPO 为 50 个候选、最多 256 epochs / boosting rounds，内部 80/20 validation，选择后不做 full-training refit。其 R² 不宜直接概括为常规树模型点回归的能力上限。（附录 B.2.3）
3. **没有直接比较用户使用的 LightGBM。** 仓库依赖包含 LightGBM 不等于主榜以 LightGBM 为 baseline；不能把本文结果写成 Causilo 已在同样市场任务击败 LightGBM。
4. **随机 split 的校准不等于未来市场校准。** 本文没有专门检验 temporal drift、重叠预测 horizon 或市场 regime。多个模型版本也只代表论文指定 checkpoint / inference 配置。
5. **coverage diagnostics 有抽样噪声。** 论文自身发现 WSC 的 split repeatability 较弱；小样本和 n 的关联可能部分包含估计精度因素，不能直接归因于模型机制。
6. **跨数据集 bootstrap 不处理 dataset family 依赖。** 论文使用 10,000 次 paired dataset-bootstrap、pointwise 95% 区间，未按数据集家族分组，也不是同时置信区间；不能把独立数据集数理解得过强。（附录 C.4、D）
7. **名称 PPD 不构成真实 Bayesian posterior 保证。** 这里主要评价模型输出 predictive distribution 的经验性质，不应仅凭 posterior 的称呼认为其条件概率已正确。

## 与 ScoringBench 的对照及可借鉴指标

ScoringBench 强调使用多种 proper scoring rules（CRPS、weighted CRPS、interval score 等）评价分布，排名依赖对哪类误差施加怎样的惩罚。TabReliBench 进一步强调：分布评分之外，应单独检查边际校准、子群 coverage 和 width。这是重点差异，不是互斥分类；ScoringBench 也包含区间评分。（[ScoringBench 论文](https://arxiv.org/abs/2603.29928)；TabReliBench §1、附录 A）

对已有 quantile 输出，建议的评估表如下。后两列属于我们的任务适配，不是本文原样采用的全部指标。

| 指标 | 回答的问题 | 在价差预测中的用法与边界 |
| --- | --- | --- |
| 逐 τ Pinball loss | 哪个分位数预测得差 | 保留上下尾和中位数，不只报平均 |
| CRPS / quantile-grid 近似 | 整体分布预测质量 | 连续定义为两倍 Pinball 的积分；稀疏 levels 的平均 loss 不能无说明地称为精确 CRPS |
| Quantile calibration curve | qτ 是否对应 τ 的观测累计比例 | 整体和同一市场状态各画一张，区分上、下尾失准；是额外诊断建议 |
| 多档 coverage + width | 是否覆盖充分、以多大区间为代价 | 同时检查 80/90/95%，分开统计上下尾漏出 |
| Interval score / WIS | 区间宽度与漏出的综合代价 | 指定 interval levels 与权重；WIS 不是本文主榜指标 |
| WSC | 搜索到的低覆盖单特征区域是否仍欠覆盖 | 可作自动扫描；不能替代共同业务子群、稀有交互和 temporal 检查 |
| PIT–KS | 全分布边际 PIT 与 uniform 的偏离 | 需足够完整的分布；稀疏 quantile 插值、目标 ties 都影响解释；时间相关下不能直接套独立样本 KS p-value |
| 同一时间块 / regime 上的 paired loss difference | TFM 的相对短板在哪里 | 同一样本比较 TFM − LightGBM，并检查多段独立行情的重复性 |

重点组合：**逐 quantile Pinball + quantile 校准曲线 + 多档 coverage–width + 按共同市场状态拆分的模型差值**。CRPS / interval score 提供综合评价，coverage curve 提供解释，不能互相替代。

## 我们的讨论：从价差预测到局部短板

### 用户观察与目标（未在此独立复核）

用户此前用相同数据、相同特征比较 Causilo quantile 与 LightGBM quantile，观察到 TFM 整体 coverage / Pinball loss 更好，但某些时间段 LightGBM 明显更好。已经针对这些坏时间段尝试过多种优化，尚未奏效；具体方法、数据、split 和结果未在本次讨论中展开。

用户的目标是：**让 TFM 在这些阶段至少接近 LightGBM，同时保留其他阶段的优势，使总体性能更强**。这不是首先寻找动态模型切换规则；组合或切换只能作为另行选择的路线。

### 这篇支持了什么，没有支持什么

- 支持的经验认识：整体强不等于局部强；点预测 / proper score 与校准不一致；通用 ensemble 或后处理不保证修复局部失败。
- 没有直接支持：哪些市场阶段 TFM 输给 LightGBM、为什么输、哪种训练方法能补齐短板。论文主要研究跨数据集差异，不是同一市场的跨时间 regime。
- “某时间段 Pinball 较差”是局部分位数预测质量问题；“该时间段 90% interval 只有 70% coverage”才是局部欠覆盖。两者都不是本文 WSC 的同义词。
- LightGBM 在相同输入上表现好，说明这些输入至少可被一种模型有效利用，不能笼统归因为信息不足；但需先确认两者训练窗口、标签可见性和损失目标可比。它不保证 TFM 经调参就能学到同样关系。

### 从失败案例到可验证解释

以下是我们的诊断建议，不是论文发现的市场机制：

| 对照 | 想区分的问题 |
| --- | --- |
| 中位数和各 quantile 的损失 / 校准 | 整体位置偏移、尺度偏小，还是单侧尾部问题？ |
| 两者完全相同历史窗口 | 树模型优势是否与更近期适配有关？ |
| LightGBM 反复使用的分裂条件及跨时间稳定性 | 是否存在阈值 / 局部交互关系？解释需验证，不能直接当因果 |
| TFM context 的时间范围、选择和相似状态样本 | 旧样本干扰或状态覆盖不足是否是可检验假设？ |
| 修复后在原优势阶段的损失变化 | 局部改善是否损害整体优势？ |

Context selection、fine-tuning、树模型学习 TFM 剩余误差/分位数修正，都只是在讨论中提出的候选方向，没有验证。最后一种会形成组合系统，不等于 TFM 本体已补齐能力；分位数修正还需处理 crossing 和联合一致性。

### 为什么要区分历史发现和后续验证

寻找坏阶段并优化是合理的。如果同一阶段被反复用来选特征、方法和参数，它已经是开发数据；需要更晚、未参与选择的时间块评估泛化。历史阶段本身也修不好，与历史修好但未来无效，是两种不同失败。

滚动 coverage 图适合定位问题，但相邻窗口不是独立证据：100 点窗口每次移动 1 点，连续两个窗口共享 99 点，一批连续漏出会留在很多窗口中。曲线可能平滑而持续下降，并不代表出现了许多独立失败事件，也不代表问题是假的。应配合不重叠时间块或独立行情事件检查。

市场实验应使用按时间的 rolling / expanding split，训练、calibration、test 按顺序分开；标签 horizon 重叠时处理切分边界泄漏。波动率、流动性、历史极端事件和输入相对 context 的异常程度，只是预测当时可获得的候选状态变量，论文没有验证这些变量能预测欠覆盖。

## 讨论补充：Quantile calibration curve 与实际校准

### 曲线是诊断图

对样本外预测定义：

```text
C_hat(τ) = mean_i[ 1{y_i <= q_hat_τ(x_i)} ]
x-axis = τ，y-axis = C_hat(τ)，理想为 y = x。
```

例如 q0.1 / q0.5 / q0.9 对应实际比例 18% / 52% / 80%，说明低分位数偏高、高分位数偏低，中位数大致正常。这是说明性例子。连续目标下可直观解释为两端向内收缩；大量 ties / 离散目标时，简单要求每个 τ 都精确相等并不总适当，需考虑 quantile 定义或随机化诊断。

在连续、严格单调分布下，quantile calibration curve 与 PIT 的经验 CDF 密切对应；有限网格、plateau 和 ties 会改变这种对应。边际曲线好同样不保证各 regime 好。画图本身不修改预测。

### 一种后处理：重新映射 quantile level

在独立 calibration 时间段估计单调映射 `C_hat`，未来希望输出 τ 分位数时，使用：

```text
q_cal_τ(x) = q_original_{C_hat^{-1}(τ)}(x)
```

如果原模型 q0.95 的观测累计比例是 90%，就可以用它作为修正的 q0.9。经验曲线可能有平坦区，实际用单调拟合和广义逆 / 插值，而不是假定严格可逆。校准只针对该 calibration 分布的边际频率，不提供未来漂移或逐 x 覆盖保证。

使用流程：训练 → 更晚的 calibration 样本外预测 → 拟合映射 → 再晚的 test。输出 levels 太稀疏、修正需求超出已有概率范围时，不能靠无依据外推补全。单一全局映射可能修复普遍过窄，也可能损害原本良好的市场状态；校准曲线改善不保证 Pinball / CRPS 改善。**这是我们的补充方法讨论，不是 TabReliBench 提出的新算法，也不是已验证的价差预测修复方案。**

## GitHub / Code Analysis

分析日期 **2026-10-16**；官方仓库 commit **`f538f3b5b230495b1fcb2dadf528a123156a3d99`**。临时 clone 在 `/private/tmp`，没有向本仓库引入第三方源码。以下来自实际阅读核心实现，未运行整套 benchmark。

### 入口与真实调用链

```text
evaluate_vanilla_tabpfn.py
  → evaluation/pipelines.py EvaluationPipeline
  → evaluation/data.py DatasetLoader / model registry / FeaturePreprocessor
  → evaluation/models.py RegressionModelRunner.fit_predict
  → 模型 fit(context) → predict(quantiles) + 独立 predict(mean)
  → ArtifactStore 保存 predictions
compute_metrics.py
  → MetricsPipeline → PPDQuantileGrid → regression / classification calculators
paper_analysis/*
  → 汇总表、score decomposition、meta-prediction、ensemble / conformal 数值分析
```

- [`evaluation/data.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/evaluation/data.py)：回归 `train_test_split` 用 `seed + 1` 生成 permutation，50/50 分开；没有按时间排序的 split。
- [`evaluation/models.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/evaluation/models.py)：默认 quantile step 为 0.005；registry 区分 point-only / quantiles，适配 `quantiles` / `alphas` 参数名。Runner 先按模型做预处理，调用 fit，再预测 quantile；用 `np.maximum.accumulate` 保证单调。点预测单独调用 `predict(output_type='mean')`，不是统一从保存网格推回均值。
- [`evaluation/ppd.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/evaluation/ppd.py)：统一 quantile 插值与 CDF 重建，重复 quantile 对应 levels 取中点；区间外 PIT 使用网格端点值。`quantile_grid_integral` 对 quantile 函数做常数尾部延伸，但 CRPS 的 loss 积分是另一函数，不补端点。

### 核心评估与论文对照

[`evaluation/metrics/regression.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/evaluation/metrics/regression.py) 中实际阅读了 `crps_per_row`、`_numeric_groups`、`_worst_conditional_coverage`、`CoverageMetric`、`WSCMetric` 和 `PinballMetric`：

- WSC 确实在前半搜索、后半验证，数值 bins 合并、类别归并、minimum support 与空组 NaN 语义符合附录 C。导出 feature name、search / eval 样本量和 reason，便于解释；不是任意方向投影或多特征 subgroup learner。
- `CoverageMetric` 在 evaluation 半计算 coverage、width、Winkler interval score；CRPS 用完整 test。`PinballMetric` 是指定 α 的上下端点 loss 及其均值，并非所有 199 个 quantile 的平均。
- 代码还计算 PIT-ECE、PIT histogram L1 和 interval score；这些存在于实现中不等于都是论文主榜指标。用于时间序列时，不能照搬默认随机 split 和 KS 显著性解释。
- 单调修正采用累计最大值，不是 sorting 或 isotonic regression；这是基准实际预测结果的一部分，复现时要保持一致。

### Baseline、conformal 与工程限制

- [`evaluation/baselines.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/evaluation/baselines.py)：`XGBoostQuantile` 路径使用 `reg:quantileerror` + `quantile_alpha=q`，内部 train/validation，修正 crossing 后再积分成 mean；与论文 quantile baseline 说明一致。不能把这个 mean 与单独训练的 MSE tree regressor 混为一谈。
- [`conformal/allocation.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/conformal/allocation.py) 区分 pooled calibration 与辅助拟合预算；[`conformal/methods/cqr.py`](https://github.com/Tianqi-Zhao/TabReliBench/blob/f538f3b5b230495b1fcb2dadf528a123156a3d99/conformal/methods/cqr.py) 的 standard signed score 为 `max(lower-y, y-upper)`，允许负阈值收缩区间，反演为 `[lower-threshold, upper+threshold]`。论文脚本的 `-standard` 与非负 score 变体不能混用。
- 官方 README：原始 held-out 数据分 A=25%、C=25%、T=50%；SCP / CalLCP / RLCP 用 A∪C 校准，SLCP / PCP / RCP 在 A 拟合、C 校准。是冻结基础预测后的预算比较，不是所有方法用完全相同数量的 calibration labels。
- 预测与 metrics 存为 pickle，可从缓存预测重算指标；`evaluate_baselines.py` 会检查已存结果，复现实验应隔离 ensemble size、sampling protocol 和版本。仓库还有 sampling/cache 协议标记。
- `requirements.txt` 指定 Python 3.12 工作流相关依赖，包含 `numpy==2.3.5`、`xgboost==3.2.0`、`tabpfn==8.0.2`、`pytabkit==1.7.3` 等。TabPFN-3.5、Causilo 等需按 README 另装指定 source revision，权重需单独获得；一个 requirements 文件不足以装齐所有模型版本。OpenML 数据需要下载，模型入口读取 checkpoint；本次未复核每个上游包内部实现。
- 官方 README 声明 original project materials 为 **CC BY-NC 4.0**；上游模型、软件和数据各有条款，见 `licenses/`。不能因为公开就假设整套代码可直接商用。README 明确只发布数值分析命令，不包含论文 plotting code。

## Insights

1. **分布质量和校准需要两套证据。** Proper score 适合比较整体预测质量，coverage / PIT / calibration curve 适合解释概率是否失准；任何单项好都不足以替代另一项。
2. **模型排名依赖问题定义和聚合。** Causilo 的校准排名与 TabPFN-3.5 的 CRPS 优势可同时成立；排名、分数中位数、特定任务的 paired loss 不应混用。
3. **局部失败可能来自位置、尺度或尾部，修复方向不同。** 必须先定位；不能把所有 TFM 不如树模型的阶段都解释成欠覆盖。
4. **找到历史失败不等于找到预测失败的特征。** 跨数据集 meta-features 对相对模型优势和 coverage 的预测有限，但这既不证明市场状态一定无用，也不证明存在可利用的稳定 regime rule。
5. **整体边际修正不能自动补齐模型能力。** 重新映射 quantile、conformal 和扩大 ensemble 各有作用，不能等同于学会 LightGBM 擅长的局部条件关系。

## 我们的观点

这篇值得收录，优先级在“评估体系与结果参考”。它给用户先前的观察提供了部分一致证据，也提供了更精确的表达：整体分布强、局部预测差、局部欠覆盖，是相关但不同的问题。

对当前 quantile 工作，最实用的是保留 Causilo / TabPFN-3.5 的对照结果，并把已有 Pinball / coverage 扩展成可解释、按状态拆分的评价。它不应被包装成时间市场上的 TFM 胜负结论，也没有证明任何一种全局后处理可以达到“保留 TFM 优势、补齐 LightGBM 强项”的目标。

## 值得继续追的问题

1. 相同时间窗口、目标和样本上，TFM 坏阶段主要是 median、scale 还是单侧 tail 失败？
2. 用共同业务子群比较两模型，结论是否与各自搜索的 WSC 一致？小群体 / 多特征交互是否被漏检？
3. 改 context、fine-tuning 或局部修正后，坏阶段改善是否在后续独立时间块重现，并保留原优势阶段性能？
4. 相同 MSE 目标训练的点回归 baseline、更多训练预算、full-data refit 是否改变本文 R² 的 TFM–baseline 差距？
5. Quantile-level remapping 与 signed CQR 对 Pinball、tail miss、width 的影响有何差异？在市场漂移下映射多久失效？
6. 以时间块 / 行情事件而非单个样本估计不确定性后，局部胜负和 coverage gap 还有多稳定？
