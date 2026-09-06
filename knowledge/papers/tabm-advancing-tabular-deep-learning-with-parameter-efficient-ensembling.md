---
type: Research Paper Review
title: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
description: "TabM 的 parameter-efficient MLP ensemble、BatchEnsemble 变体与 weak-individual/strong-ensemble 现象分析，重点记录并行成员训练、ensemble-aware early stopping、官方实现以及可推广的 ensemble 机制。"
resource: https://arxiv.org/abs/2410.24210
tags: [tabular-deep-learning, ensemble, mlp, batchensemble, parameter-efficient-ensembling]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2410.24210
    title: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
  - id: code
    resource: https://github.com/yandex-research/tabm
    title: yandex-research/tabm
---

# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

## Links

- Paper: [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
- Official code: [yandex-research/tabm](https://github.com/yandex-research/tabm)
- ICLR 2025.
- 本轮复核时官方仓库 `main` HEAD 为 `28e47ae301c92ec37787dde1ce923a0793f405b4`。

## 一句话结论

TabM 的核心不是复杂的新 tabular backbone，而是把 deep ensemble 变成一个带显式 member 维度、绝大多数权重共享、一次训练同时得到多份预测的 MLP。成员同时训练使 early stopping / model selection 可以直接围绕最终 ensemble；BatchEnsemble 式低成本 member-specific modulation 又能制造足够的预测差异。

论文最有价值的 empirical finding 是 **weak individual / strong ensemble**：单个 TabM member 并不比普通 MLP 更强，但平均预测明显更强。这更接近“共享表示 + 互补误差 + ensemble-aware model selection”的共同结果，而不是内部偷偷训练出多个更强单模型。

## 文章摘要

TabM 基于 BatchEnsemble。普通线性层共享大矩阵 `W`，第 `i` 个 member 只增加输入/输出 scaling 向量 `r_i`、`s_i` 和 bias：

```text
l_i(x_i) = s_i ⊙ (W (r_i ⊙ x_i)) + b_i
```

所有 member 同时前向/反向；每个 member 独立计算任务 loss，再对 member losses 求平均。推理时 regression 平均预测，classification 先将每个 member logits 转成概率再平均。

## 方法拆解

### 显式 ensemble 维度

官方 package 用 `EnsembleView` 把 `(B,D)` copy-free 扩展成 `(B,K,D)`，K 默认 32。后续层都把 K 当成显式 member 维度，因此模型一次 forward 直接产生 K 份 prediction。

### BatchEnsemble layer

官方 `LinearBatchEnsemble` 中：

- shared `weight`: `(out_features, in_features)`；
- member-specific `r`: `(K, in_features)`；
- member-specific `s`: `(K, out_features)`；
- member-specific bias: `(K, out_features)`。

forward 等价于：

```python
x = x * r
x = x @ weight.T
x = x * s
x = x + bias
```

参数高效性来自最昂贵的矩阵 W 共享，而成员额外参数主要是 O(D) scaling vectors。

### TabM / TabMmini / TabMpacked

- `TabMpacked`：K 个完全独立 MLP 参数，只是打包并行训练；用于隔离“同时训练 + ensemble-aware early stopping”的收益。
- `TabMmini`：只在输入端保留一次 member-specific scaling，后续是普通共享 MLP；它仍然很强，说明大量 adapters 不是必要条件。
- `TabM`：每个 MLP linear block 使用 BatchEnsemble，但采用特殊初始化：第一层 input scaling 随机，第一层 output scaling 为 1，后续 multiplicative scalings 都从 1 开始。

### 最终 prediction head 完全独立

这是官方实现里容易被“权重共享”概括掉的细节。backbone 大量共享参数，但最后用 `LinearEnsemble`，其 weight 为 `(K,d_in,d_out)`，因此每个 member 都有完全独立的 prediction head，保留了形成不同 residual / decision boundary 的自由度。

### loss 不是 ensemble prediction 的 loss

官方训练语义是：

```text
L = mean_i L(y_hat_i, y)
```

而不是：

```text
L(mean_i y_hat_i, y)
```

这保证每个 member 仍需独立解决完整任务，而不是允许弱 member 完全依赖其他成员补偿。

### shared batch 与 independent batch

当前 package 支持共享 `(B,D)` batch，也支持 `(B,K,D)` 让每个 member 使用自己的完整 batch。当前官方更推荐 shared batch 以简化和提效；独立 shuffle/batch sequence 可以增加 optimization noise diversity，但不是方法成立的必要条件。

## 关键实验结果

论文覆盖 46 个 datasets：28 regression、18 classification；37 random splits、9 domain-aware splits。

- independent `MLP×32` 相对普通 MLP约 `0.96% ± 1.6%` relative improvement；
- `TabMpacked` 约 `1.42% ± 2.3%`，支持同时训练和 ensemble-aware early stopping 本身有收益；
- `TabMnaive` 约 `1.86% ± 2.4%`；
- 默认 `TabM` 约 `2.15% ± 2.8%`。

最好单 member `TabM[B]` 相对普通 MLP约 `-0.06% ± 1.8%`，即大体持平甚至略差；但 32-member ensemble 明显更强，直接支撑 weak-individual/strong-ensemble 现象。

训练后 greedy member selection `TabM[G]` 约 `2.02% ± 2.6%`，从 32 个成员平均保留 `8.8 ± 6.6` 个，说明可以训练时用较大 K 产生 diversity、推理时再裁剪。

## 为什么可能形成 weak individual / strong ensemble

这是我们的机制解释，不是论文已经严格证明的理论。

共享 backbone W 接收所有成员梯度平均：

```text
∇W L = (1/K) Σ_i ∇W L_i
```

不同成员的局部过拟合方向若不一致，较稳定、跨成员一致的方向更容易保留；与此同时 `r_i/s_i`、独立 prediction head、dropout 和 batch noise 允许成员形成不同 residual error。可粗略写成：

```text
y_hat_i = f_shared(x) + epsilon_i(x)
```

如果误差相关性不高，平均后的 `epsilon` 会下降。

另一个关键因素是 ensemble-aware early stopping：训练可能选择一个“单成员已不同程度过拟合，但平均预测仍最优”的 checkpoint，这与每个独立模型各自 early-stop 的传统 deep ensemble 不完全相同。

## GitHub / Code Analysis

我们实际核对过官方代码的核心路径：

- `EnsembleView`：`(B,D)` copy-free 扩成 `(B,K,D)`；
- `LinearBatchEnsemble`：共享矩阵 + member-specific `r/s/bias`；
- `MLPBackboneBatchEnsemble`：默认 TabM 初始化；
- `MLPBackboneMiniEnsemble`：只保留输入侧 member-specific affine；
- `make_tabm_backbone`：映射 `tabm` / `tabm-mini` / `tabm-packed`；
- `TabM`：共享 preprocessing，显式 ensemble view，backbone 后接独立 `LinearEnsemble` prediction heads；
- `example.ipynb`：成员独立 loss、ensemble validation early stopping 和 inference aggregation。

从实现上看，TabM 主要是在 PyTorch tensor shape 上把 member 维度做成一等公民，再用部分共享 linear layers 实现高效 population。

## Insights

1. `TabMpacked > MLP×k` 说明 ensemble-aware training protocol 本身就是方法的一部分。
2. parameter sharing 的价值不仅是省参数，也可能是跨成员 regularization / information sharing。
3. 不应以单 member accuracy 判断其 ensemble 价值。
4. `TabMmini` 表明很低成本的 member-specific perturbation 就可能制造有用 diversity。
5. 训练目标和选择目标可以不同：成员用 individual loss 训练，ensemble 用 validation performance 选择 checkpoint。

## 我们的观点

TabM 更偏模型 / 训练机制贡献。最值得推广的不是“BatchEnsemble 必须照搬”，而是：显式并行多个完整预测函数、尽量共享昂贵计算同时保留 member-specific freedom、用最终 ensemble 驱动 early stopping / selection。

推广到 tree / heterogeneous model 时不一定继续共享 W；真正应该保留的是成员独立任务 loss、diversity 和 ensemble-aware selection。

## 值得继续追的问题

- gradient similarity / error correlation 能否解释 weak-individual/strong-ensemble；
- independent batches、dropout、adapter initialization 各自贡献多少 diversity；
- 独立 prediction head 是否是关键自由度；
- 能否推广到 ResNet、GCN、ConvNet 等规则 backbone；
- 固定单图 node regression 能否共享 graph topology / sparse propagation，而 pack hidden states 和参数；
- member selection 能否直接优化 marginal ensemble gain / error complementarity。
