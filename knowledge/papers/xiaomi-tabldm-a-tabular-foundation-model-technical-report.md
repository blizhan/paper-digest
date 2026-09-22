---
type: Research Paper Review
title: "Xiaomi-TabLDM: A Tabular Foundation Model Technical Report"
description: "Xiaomi-TabLDM 的 SCM prior、三段式 ICL 架构与分位数回归；核对官方仓库的单视图/等权分位数集成及 NNLS 仅输出点预测的实际边界，并与 TabICLv2、Mitra-v2、LimiX-2 对照。"
resource: https://arxiv.org/abs/2609.03880
tags: [tabular-foundation-model, synthetic-prior, scm, in-context-learning, probabilistic-regression, quantile-regression, ensemble]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2609.03880
    title: "Xiaomi-TabLDM: A Tabular Foundation Model Technical Report"
  - id: code
    resource: https://github.com/xiaomi-research/xiaomi-tabldm
    title: "xiaomi-research/xiaomi-tabldm (official inference code)"
---

# Xiaomi-TabLDM: A Tabular Foundation Model Technical Report

## Links 与阅读范围

- 论文：[arXiv:2609.03880](https://arxiv.org/abs/2609.03880) · [PDF](https://arxiv.org/pdf/2609.03880)
- 官方：[GitHub](https://github.com/xiaomi-research/xiaomi-tabldm) · [Hugging Face](https://huggingface.co/occams/Xiaomi-TabLDM)
- **源码核对版本**：[`3335755ad538b80614f000226e2856b543a0273d`](https://github.com/xiaomi-research/xiaomi-tabldm/tree/3335755ad538b80614f000226e2856b543a0273d)。以下「仓库实际怎么做」仅针对该 commit；论文实验结果是作者报告，**没有独立运行权重或重现 benchmark / 校准实验**。
- 关联笔记：[TabICLv2](tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model.md) · [Mitra-v2](mitra-v2-technical-report.md) · [LimiX-2](limix-2-a-contextual-mechanism-network-towards-general-structured-data-intelligence.md) · [O'Prior](shaping-the-prior-how-synthetic-task-distributions-determine-tabular-foundation-model-quality.md)。

## 一句话结论

Xiaomi-TabLDM 沿用 **SCM 合成任务预训练 → 列编码 → 行聚合 → dataset-level ICL** 的表格基础模型范式；在接近 TabICLv2 的分层架构上引入双流特征分组、轻量 AttnRes 与 sparse MoE，并在回归端直接预测 **999 个分位数**。论文的回归 benchmark 表现与**概率预测是否校准**是两个不同问题：官方普通推理能返回分位数，但 **NNLS 增强模式的公开 `predict()` 当前只输出 `mean`，没有提供 NNLS 后的分位数分布**。（论文 §2–4；官方 `regressor.py`）

## 1. 方法：哪些部分是共通范式，哪些是具体改动

```text
sample dataset properties → Cauchy DAG / SCM → generate (X, y)
        ↓                         synthetic ICL tasks
column-wise: dual-stream 3-feature grouping + Set Transformer
        ↓
row-wise: Transformer + four CLS tokens + lightweight AttnRes
        ↓
dataset-wise ICL: labeled context + unlabeled query; last 8/24 blocks use sparse MoE
        ↓
classifier: class logits       regressor: 999 conditional quantiles
        ↓
optional test-time views / ensemble / validation-fitted NNLS (point prediction)
```

**架构细节（论文 §2.1、附录 A.1）**：Column 侧每次用循环偏移将三个特征编为一组，第一流偏移沿用 TabICLv2 的 `(1,2,4)`，第二流随列宽自适应扩大跨度；经 induced-attention / Set Transformer 后相加。Row 侧通过 4 个 CLS token 汇聚为 512 维行表示。ICL 侧为 24 个 Transformer blocks，最后 8 层将部分 FFN 换成稀疏 MoE：每层 2 个 routed experts、每个 token 选择 1 个，另有 1 个 shared expert。AttnRes 让部分层按输入自适应聚合先前的 block 表示，而非只使用普通固定残差。分类模型约 70.08M 总参数 / 61.67M 激活参数；回归约 71.08M / 62.68M。

**SCM prior（论文 §2.3、Figure 5）**：随机生成表格形状、变量类型与类别数 → 采样 TabICLv2 风格的 Cauchy DAG → 让根节点与非根节点通过随机分布、函数及父节点聚合产生潜在表示，注入噪声 → 选择观测维度并转成数值或类别特征及目标 → 检查 `X` 和 `y` 是否共享祖先，并用 ExtraTrees 相对常量预测器的表现过滤无信息任务 → 构造 context/query。**这里的 SCM 是训练数据生成先验，不是部署时显式输出 DAG 或估计干预效应。**

**预训练（论文 §2.2）**：分类/回归独立训练；分类用 cross-entropy，回归用 999-quantile pinball loss。三阶段依次为：1,024 行任务上的分类 500K / 回归 300K steps；400–10,240 行任务 40K steps，开启 MoE；400–60,000 行任务 10K steps，适配长上下文。论文报告使用 8×A100 80 GB、Muon、余弦学习率；该规模是训练安排，不代表当前仓库已包含可重训的完整数据生成/预训练脚本。

**公开边界**：当前仓库可见 PyTorch 模型、推理/预处理、sklearn 封装、checkpoint 下载及推理教程；本次检查的文件列表中**未见完整 SCM 生成器、三阶段预训练入口及其端到端复现脚本**。研究可修改的通用 prior 时，参照 [TabICLv2 公开生成器](https://github.com/soda-inria/tabicl)，不能把两者默认当成逐行相同的合成数据实现。

## 2. 论文实验究竟支持什么

| 论文评测范围 | Xiaomi-TabLDM 的作者报告 | 解读条件 |
| --- | --- | --- |
| OpenML-CTR23：33 个回归数据集 | 平均 rank **3.03** | 任务内相对排序；不是概率预测指标（论文 Figure 2）。 |
| TALENT：回归子集 | 平均 rank **4.03** | 论文 Figure 3；不能与其他 benchmark 的 Elo 直接横比。 |
| TabArena：51 数据集、816 个任务单元（全任务） | Elo **1659** | 论文 Table 11；同表还报告每 1K 样本推理 **4.18 s**。 |
| TabArena：13 个回归数据集 | Elo **1900** | 论文 Table 12；每 1K 样本训练 **6.99 s**、推理 **3.12 s**；同表 TabFM 为 Elo **2019**、训练 **38.85 s**、推理 **9.67 s**。 |

论文的这些指标说明所测表格任务上的**点预测/分类表现**，并不直接回答概率分布是否可靠。正文及附录没有找到针对外部真实数据的 quantile coverage、CRPS、分位数校准曲线，也缺少能单独归因双流、AttnRes、MoE、prior 修改及 NNLS 的完整定量模块消融；AttnRes 的稳定训练现象是作者在 §2.2 中的**定性描述**，不能替代同预算的数值对照。本文不把作者的 benchmark 点估计解释成其他时间切分或分布漂移场景的保证。

## 3. 源码：基础分位数、普通 Ensemble、NNLS 是三条不同路径

| 路径 | 参数 / 入口 | 实际返回内容 | 关键源码 |
| --- | --- | --- | --- |
| 单视图基础模型 | `n_estimators=1, enhance_candidates=False` | `mean` / 指定 `quantiles` / 全部 `raw_quantiles` | [`regressor.py` `predict()`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1515-L1604) |
| 普通多视图集成（默认） | `n_estimators=8, enhance_candidates=False` | 每个视图分别输出分位数；对相同水平的分位数**逐点算术平均** | [同文件 `#L1587-L1600`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1587-L1600) |
| 增强候选 + NNLS | `enhance_candidates=True, validation=True` | **只支持 `mean` 点预测**：多视图/变换 + 验证或 OOF 权重 | [同文件 `#L1606-L1773`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1606-L1773) |

### 3.1 从模型 raw quantiles 到 CDF

[`tabldm/_model/tabldm.py#L547-L628`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_model/tabldm.py#L547-L628) 在回归时先产生 `(batch, n_query, 999)`，随后用 [`QuantileToDistribution`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_model/quantile_dist.py#L1509-L1569) 把它转成分布对象：默认对 crossing **排序修正**，在相邻分位数间分段插值，并用默认指数尾部外推；分布对象有 `icdf()`、`cdf()`、`crps()` 等方法。`raw_quantiles` 返回的是**已经过单调性修正**的分位数，而不是未经处理的 head logits。默认 999 个水平在 `(0,1)` 内等距排列；尾部外推是**分布构造假设**，不等于尾部概率已被真实数据校准。

公开 sklearn 调用方式（[`tutorials/regression_quantiles.py`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tutorials/regression_quantiles.py)）：

```python
from tabldm import TabLDMRegressor

reg = TabLDMRegressor(n_estimators=1, enhance_candidates=False)
reg.fit(X_train, y_train)  # 加载权重、保存有标签 context；不做下游参数更新
q999 = reg.predict(X_test, output_type="raw_quantiles")  # (n_test, 999)
q10_50_90 = reg.predict(X_test, output_type="quantiles", alphas=[0.1, 0.5, 0.9])
```

`fit()` 还对 `y_train` 拟合 `StandardScaler`，输出在 `predict()` 内反变换回原始目标量纲（[`regressor.py#L1174-L1185`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1174-L1185)、[`#L1587-L1600`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1587-L1600)）。

### 3.2 普通集成：平均同级分位数，而非混合 CDF

[`EnsembleGenerator`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/preprocessing.py#L1210-L1270) 生成特征排列和归一化视图；普通回归默认使用 `none` / `power` 两种归一化候选。每个视图各自预测并还原 `y` 的量纲，`predict()` 对形状为 `(n_estimators, n_test, 999)` 的数组沿成员轴 `np.mean(arr, axis=0)`。即

\[
Q_{\mathrm{avg}}(\tau)=\frac{1}{M}\sum_{m=1}^{M}Q_m(\tau).
\]

这是**quantile averaging**，不是先混合 `F_m` 得到 `F_mix(y)=Σ_m w_m F_m(y)` 再取其逆 CDF。两者一般不等价，对多峰、不确定性分歧、区间宽度和 coverage 的影响也可能不同。普通集成没有基于验证集再拟合一层分位数校准器。

### 3.3 增强模式：NNLS 优化点预测，不能直接当成概率集成

[`regressor.py#L1143-L1493`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1143-L1493) 在 `enhance_candidates=True` 下生成主视图、额外量化归一化视图、SVD/交互等候选，并可对高峰度目标使用额外目标变换；通过 K-Fold OOF 或留出验证集收集**各候选的 `mean` 点预测**，使用 `scipy.optimize.nnls(pred_matrix.T, y)` 拟合非负权重，之后归一化。该 NNLS 最小化的是**点预测平方误差**，不是 pinball loss / CRPS / coverage loss。默认 `foundation_rate=0.25`，最终点预测等于 `0.75 × NNLS 加权点预测 + 0.25 × 等权点预测`（[`#L1712-L1769`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1712-L1769)）。

最重要的源码边界是 [`predict()` 中的 `output_type = ["mean"]  # enhanced path only supports mean`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/3335755ad538b80614f000226e2856b543a0273d/tabldm/_sklearn/regressor.py#L1604-L1608)：传入 `output_type="quantiles"` 或 `"raw_quantiles"` **仍会进入强制 `mean` 的增强分支**，不会提供 NNLS 后的分位数。文件末尾虽然保留了针对三维数组加权的代码分支，但当前增强推理入口没有把分位数传进该分支；不能据此宣称官方已经实现 NNLS 概率集成。

## 4. 与我们此前讨论的模型放在一起

| 模型 | 合成训练先验 | 表格信息交互 | 回归分布表示 / 目标 | 下游默认推理需区分 |
| --- | --- | --- | --- | --- |
| Xiaomi-TabLDM | Cauchy DAG / SCM；额外函数与过滤 | Column → Row → Dataset ICL；dual-stream、AttnRes、MoE | **999 quantiles / pinball** | 普通分位数平均 vs 增强 NNLS **仅点预测** |
| [TabICLv2](tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model.md) | GraphSCM 等 synthetic prior | Column → Row → Dataset ICL | quantile regression | 推理也有特征变换 / ensemble |
| [Mitra-v2](mitra-v2-technical-report.md) | SCM + Hybrid SCM + tree priors | cell-wise 2D，行/列交替 attention | **1,000-bin cross-entropy**，非 pinball | TabArena 主结果含 full FT + bagging，非纯 zero-shot |
| [LimiX-2](limix-2-a-contextual-mechanism-network-towards-general-structured-data-intelligence.md) | 扩展 SCM | 样本/特征双轴 attention，特征与任务分路 | 分桶回归；另有 CCMM 特征重建 | 推理 ensemble、任务目标均与 Xiaomi 不同 |

共同范式是「对海量合成表格任务做 ICL 预训练，再在真实表格上评测」，但**不都是同一种回归 loss，也不是同一套部署系统**。尤其不能直接比较各论文在不同候选池、调参预算与切分协议下的 Elo 点估计。Xiaomi 与 TabICLv2 的网络分层和分位数目标更接近；LimiX-2 的 CCMM 另外训练被遮盖变量的条件重建，不能简化成只预测固定 `y`。

## 5. 概率预测与价差应用：尚未被论文回答的问题

我们原来问：「只用基础模型的 999 个分位数是否已经可靠？增加 Test-Time Scaling 后分布与校准怎样变化？」**源码现在只能回答“怎样得到分位数／怎样集成”，不能回答“真实校准质量如何”。** 分位数 head、排序修正、可计算 CDF/CRPS 都不自动保证校准；NNLS 点预测增强也没有现成的最终概率分布。因此不应把 TabArena 回归 Elo、RMSE 或 `predict()` 输出直接等同于 `P(ΔP>0)` / `P(ΔP>50)` 的可靠性。

值得验证的四条**实验方案（不是官方已报告的结果）**：单视图 `raw_quantiles`；普通多视图等权 quantile averaging；研究性变体——按视图 CDF 做等权 mixture；研究性变体——以独立校准集用 proper scoring rule 学习概率混合权重，并与官方 **仅点预测** NNLS 对照。统一 context/特征/预算，使用滚动时间切分且隔离权重拟合、概率校准和最终测试，报告 pinball、CRPS、80/90/95% coverage 与宽度、分层 reliability、价差阈值 Brier Score，单独核查尖峰尾部及分布漂移。

**实现注意**：官方普通 `raw_quantiles` 已在原始 `y` 量纲，避免重复反标准化；构造 mixture 时应先为每个视图形成合法 CDF，再混合，**不能把平均分位数当作混合分布的分位数**；若测试视图曾参与选权重或校准，必须防止数据泄漏。

## 后续值得追的证据

1. 用同一真实数据、固定随机种子测 `n_estimators=1` 与普通等权分位数集成的 coverage / CRPS / 分层阈值概率，而不把 `enhance_candidates=True` 误当成分位数输出。
2. 在同样预训练规模与推理预算下分别消融 dual-stream、AttnRes、MoE、先验生成器与特征集成，区分 backbone 与 test-time views 的收益。
3. 如果需要通用 SCM 可修改生成器或 domain-aware prior，优先核对 TabICLv2 开源预训练代码；当前 Xiaomi 仓库不能直接复现其完整 SCM 预训练过程。
