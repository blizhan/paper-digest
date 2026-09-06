---
type: Research Paper Review
title: "TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning"
description: "TabPack 的 packed hyperparameter ensemble、向量化模型/优化器与 online ensemble selection 分析，重点讨论它如何把传统 HPO 转化为一次 population training，以及向 ResNet、GCN、Conv 等 backbone 推广的可能性。"
resource: https://arxiv.org/abs/2607.05380
tags: [tabular-deep-learning, ensemble, hyperparameter-optimization, population-training, packed-model, automl]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2607.05380
    title: "TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning"
  - id: code
    resource: https://github.com/yandex-research/tabpack
    title: yandex-research/tabpack
---

# TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning

## Links

- Paper: [arXiv:2607.05380](https://arxiv.org/abs/2607.05380)
- Official code: [yandex-research/tabpack](https://github.com/yandex-research/tabpack)
- ICML 2026.
- 本轮复核时官方仓库 `main` HEAD 为 `05a89e21b955f12de84889d662e15ca534019aaa`。

## 一句话结论

TabPack 可以理解成把传统 `sample HP -> train -> repeat -> select` 重构成 **一次 packed heterogeneous population training + online checkpoint/ensemble selection**。它最重要的贡献不是证明“异构 hyperparameters 天然比同 HP ensemble 更强”，而是把原本昂贵的 HPO 和 ensemble search 变成高吞吐的一次训练流程。

## 文章摘要

TabPack 在一次 run 中采样并训练许多不同 MLP 配置。成员可以拥有不同 depth、width、dropout、embedding、optimizer、learning rate、weight decay 等，但成员之间不共享权重。

方法把模型参数、hidden states 和 optimizer states 沿 pack dimension 堆叠，再用 batched tensor operations 并行训练；训练过程中持续收集当前成员和历史 checkpoint 的 validation predictions，通过 greedy 等算法在线重建 ensemble。

## 方法拆解

### model pack：逻辑独立，物理堆叠

普通一层 `W: (D_out,D_in)`，K 个成员变为 `W_pack: (K,D_out_max,D_in_max)`。官方 `LinearPack` 接收 `(pack_size,batch_size,in_features)`，核心使用 `torch.bmm` / `torch.baddbmm`；成员 width 不同时，对超出真实 `out_features` 的部分做 member-specific output mask。

所以本质是 K 个独立 W stack 成 `ParameterPack` 后交给 GPU batched matmul，而不是像 TabM 那样共享大矩阵。

### optimizer pack

官方 `OptimizerPack` 表示一组可拥有不同 learning rate、weight decay 等 hyperparameters 的 optimizers。`AdamWPack` 的 `lr/beta1/beta2/eps/weight_decay` 都支持 scalar 或 per-member list，通过 broadcasting 一次更新 packed parameter tensor。scheduler 同理，只需每个 step 生成 member-specific `lr_pack[K]`。

### Muon

官方 optimizer code 引入 Muon，训练代码也对 packed linear layer 构造 Muon scale。Muon 比 AdamW 更难 pack，因为包含矩阵级变换；成员 width 不同时必须让 padding / inactive submatrix 不污染矩阵运算。因此 Muon 可以套同一设计，但需要 shape/mask-aware packed implementation。

### heterogeneous depth / width

不同 width 用最大宽度 + mask；不同 depth 让浅成员在更深 block skip / identity。最重要的实现条件是：不同成员最好能表达为“同一个 tensor program + member-specific mask”。计算图差异越大，packing 的 GPU 效率越容易下降。

### online ensemble

TabPack 在训练过程中维护 candidate pool。候选可来自 running member 的 latest/best checkpoint、finished member final checkpoint，也可把当前 ensemble 加回候选池。官方 `OnlineEnsemble` 保存 ids、steps、predictions、weights、score 和独立 patience；同一个 base model 在不同 epoch 的 checkpoint 也可以成为不同候选。

成员和 ensemble 有各自的 stopping 逻辑。官方 `compute_stop_pack_idx` 会依据每个 member 的 patience / epoch budget 独立停止，而 `OnlineEnsemble` 则持续从不同 member、不同 epoch 的 validation predictions 中重建最终组合。这意味着 TabPack 搜索的实际单位并不只是“模型配置”，而是“配置 × checkpoint”。

## 关键实验结果

论文主结果的 17-dataset Figure 6 中：

- `TabPack†` 平均 rank `2.6 ± 1.4`，A100 runtime `2.4h`；
- 在 MacBook M4 Pro 上，TabPack 仍为 rank `2.6 ± 1.3`，runtime `22.6h`；
- tuned `TabM†` 为 rank `2.8 ± 1.0`，HPO runtime `47.2h`；
- `MLP†HPE` 为 rank `3.6 ± 1.4`，runtime `3.9h`；
- `TabPack†Offline` 为 rank `3.9 ± 1.4`，runtime `2.6h`。

这里最重要的 signal 是效率：TabPack 用一次 packed run 把大量 HP population 和 ensemble selection 一起做掉，达到接近 tuned TabM 的精度，同时显著减少完整 HPO 所需时间。

### heterogeneous HP 并不天然更强

论文 Table 1 是另一个只有三种设置参与的 rank 对比，不能和 Figure 6 的全方法 rank 混用：

- random heterogeneous HP: `1.6 ± 0.6`；
- `DiverseWidths`: `1.6 ± 0.5`；
- `SameHP`: `1.4 ± 0.7`。

`SameHP` 反而略好。因此论文更有力地证明的是：**heterogeneous HP 让我们可以不先做昂贵 tuning，就获得一个覆盖多种 inductive bias 的 population，再从中选 ensemble**；它没有证明随机异构 HP 会提高 ensemble 的性能上限。

### online selection 本身很重要

TabPack 每个 epoch 都可以重建 ensemble。candidate pool 可以同时保留当前模型、历史 checkpoint 和先前 ensemble；同一个 base model 在不同 epoch 的 checkpoint 可以共存。论文默认最大 ensemble size 为 32，最终 ensemble 平均大小约 `16 ± 9`；member patience 约 16，ensemble patience 约 32。

`TabPack†Offline` 明显弱于 online 版本，说明“训练后统一选一次”并不能完全复现其收益。这里的优化对象已经从单模型 best checkpoint 变成了一个动态 checkpoint population。

### 大规模 temporal datasets

在 5 个 1M+ rows 的 temporal datasets 上，论文报告：

| Dataset | TabPack | TabM | Better |
| --- | ---: | ---: | --- |
| Homecredit | 0.8693 | 0.8737 | TabM |
| Delivery ETA | 0.5404 | 0.5410 | TabPack（lower better） |
| Weather | 1.3899 | 1.4065 | TabPack |
| Maps Routing | 0.1572 | 0.1595 | TabPack |
| Cooking Time | 0.4758 | 0.4780 | TabPack |

TabPack 赢 4/5，但论文中的 baseline HP 是在 subsampled versions 上调过的，因此更稳妥的结论是：packed population 在大规模 temporal setting 也保持竞争力，而不是“异构 packing 在结构上普遍优于 tuned 单一架构”。

### memory 不是免费的

论文给出的 peak memory 对比：

| Pack size | TabPack | TabM |
| --- | ---: | ---: |
| 32 | 3.0 GB | 2.1 GB |
| 64 | 5.8 GB | 4.0 GB |
| 128 | 11.3 GB | 7.7 GB |

TabPack 通过 batch tensor program 提高吞吐，但它保留每个 member 的独立参数和 optimizer state，因此显存会比参数共享更强的 TabM 高。

## GitHub / Code Analysis

我们实际核对了官方代码的核心实现，对应本轮 `main` HEAD `05a89e21b955f12de84889d662e15ca534019aaa`。

### `src/project/nn.py`

- `ParameterPack`：把多个逻辑模型参数组织为一个 packed parameter；
- `LinearPack`：输入显式为 `(pack_size, batch_size, in_features)`，使用 `torch.bmm` / `torch.baddbmm`；
- 不同 member 的 `out_features` 通过 output masks 表达。

这说明“不同 width”并不是 Python 层跑 K 个 module，而是用 maximal shape + mask 映射到一个 tensor program。

### `src/project/optim.py`

- `OptimizerPack`：一组逻辑独立 optimizer 的统一接口；
- `AdamWPack`：`lr`、`beta1`、`beta2`、`eps`、`weight_decay` 都可为 per-member value；
- broadcasting helper 把 member-level HP tensor 扩到 parameter dimensions 后，一次执行更新；
- 文件中还引入 `vendor.muon`。

因此我们前面讨论的“lr / scheduler / W 都不一样为什么还能同时优化”，答案在实现上很直接：**W 和 optimizer state 本来就有 pack dimension；lr/scheduler 也是沿同一 member dimension 的向量。一次 tensor update 并不要求所有成员 HP 相同。**

Muon 也不是原理上不能 pack，只是它的矩阵级更新比逐元素 AdamW 更敏感于 padded shape；若 width 不同，需要保证 inactive rows/columns 不进入矩阵变换或统计量。

### `src/project/tabpack.py`

- `compute_stop_pack_idx`：成员可以按各自 patience / epoch budget 独立结束；
- `OnlineEnsemble`：维护 ids、steps、predictions、weights、score、patience；
- 支持 `latest` / `best` / `final` candidate update modes；
- ensemble 可以从历史 candidate checkpoints 反复重建。

这里验证了 TabPack 并不是“同步跑一批 HP 后最后取平均”，而是把 checkpoint selection 和 ensemble construction 直接放进训练循环。

### 工程限制

packing 不是自动获得的。每出现新的 layer / optimizer，如果其运算不能直接用现有 batched primitive 表达，就需要专门实现 packed version。架构差异越大，mask、padding、control flow 和内存浪费越严重。

论文也承认反复使用同一 validation set 做 online selection 可能引入 validation overfitting。TabArena 的 small/IID setting 中，作者后来对 classification greedy scoring 改用 training loss 并报告更好效果，这进一步说明 selection protocol 本身是需要单独研究的变量。

## 与 TabM 的关系

两篇工作的共性都是把 “ensemble member” 变成 tensor shape 中的一等公民，但共享程度完全不同：

```text
TabM
  member dimension K
  ├─ shared expensive backbone weights
  ├─ cheap member-specific modulation / heads
  └─ goal: parameter-efficient ensemble

TabPack
  member dimension K
  ├─ independent model parameters
  ├─ independent optimizer HP / states
  ├─ max-shape + masks unify execution
  └─ goal: efficient heterogeneous population + HPO + ensemble search
```

TabM 更像“同一架构里做高效 diversity”；TabPack 更像“同一 tensor program 里打包多个不同配置”。前者强调参数共享，后者强调计算图统一。

这也解释了两种方法的适用边界：如果成员差异主要是轻量 modulation，TabM 式共享更省；如果想同时探索 width、depth、dropout、optimizer、lr 等 HP，TabPack 式独立参数更自然。

## 从我们的讨论得到的推广方向

这里属于基于论文与代码的研究推断，不是作者已经验证的结论。

### 一个通用判断标准

最关键的不是“模型是不是 MLP”，而是能否写成：

```text
same tensor program + member-specific parameters/masks
```

只要多个模型可以被映射到同一批量算子和有限的 mask/control logic，就有机会得到 packing 的吞吐收益。

### ResMLP / MLP family

这是最容易迁移的方向。Linear、activation、dropout、residual 都能保持相同执行路径；width/depth 用 maximal shape + mask / block skip 处理即可。它最适合先做一个非 tabular 的概念验证。

### ResNet / CNN / Conv family

这是我们认为很有希望、也方便验证的第二类。

- independent conv kernels 可以沿 pack/group dimension 存储；
- width 用 channel mask；
- depth 用 residual block skip；
- 不同 kernel size 理论上可以嵌入 maximal kernel 后把外围权重置零。

但这里需要实测 kernel padding 和 grouped/packed conv 是否真能转化成 wall-clock gain。理论可表达不等于 GPU kernel 一定高效。

### GCN / GNN：固定单图多节点回归尤其适合

对于我们讨论的“同一张固定图、多节点回归”，graph topology 完全共享，天然满足大量公共结构：

```text
H_pack: (K, N, D)
W_pack: (K, D_in, D_out)

H'_i = sigma(A H_i W_i)
```

`A` / `edge_index` 可以共享，而每个 member 拥有自己的 hidden state 和 weights。若 sparse propagation 能围绕相同 topology 做 batched execution，就可能同时复用图结构开销和 kernel launch。

建议顺序是 `GCN / GIN / GraphSAGE` 先于 GAT：attention 和 neighbor sampling 会增加 member-specific dynamic behavior，packing 难度更高。

node split 也必须认真设计。随机 node split 往往是 transductive，message passing 可以跨 train/valid/test 节点传播；如果目标是严格 generalization，应考虑 temporal / spatial / community / inductive split，而不是把随机 node split 的结果直接解释成独立样本泛化。

### Transformer / SSM

理论上同样可以 pack，但 head count、hidden width、normalization、KV/state dimension 等变量会制造更多 shape/mask complexity。它们不是不能做，而是比 ResMLP / CNN / GCN 更适合作为后续验证。

我们的优先级可以写成：

```text
ResMLP → ResNet/CNN → GCN/GIN → TCN → Transformer → Mamba/SSM
```

### 研究 framing 应该从 HPO 再往前一步

如果推广到这些 backbone，最有研究价值的 framing 不是“把很多任意异构模型塞进一个 batch”，而是：

> Packed Architecture Ensemble within a model family

目标不只是找最强单模型，而是低成本搜索一组 **complementary prediction functions**。因此 selection criterion 可以从 individual validation rank 进一步改成 marginal ensemble gain、residual correlation、error complementarity 等直接面向 ensemble 的指标。

## Insights

1. TabPack 真正把 HPO 变成了 population training 问题，而不是把单次 trial 再加速一点。
2. “不同 HP 还能一次优化”依赖的是 pack dimension，而不是所有成员共享 optimizer hyperparameters。
3. heterogeneous HP 的主要价值目前是减少 tuning 成本，并没有证据证明其 ensemble ceiling 必然更高。
4. online checkpoint selection 让“epoch”也成为 ensemble diversity 的来源；一个 base model 可以贡献多个不同训练阶段的预测函数。
5. packing 自身不带来统计上的提升，它只是让更大的 population、architecture search 和 ensemble search 变得便宜。
6. 可推广性的核心约束是 tensor-program compatibility，而不是 backbone 名称。

## 我们的观点

TabPack 比单纯的“并行 HPO 工程”更有研究味，因为它改变了搜索单位：一次 run 里同时探索模型配置、optimizer 配置、训练 checkpoint 和最终 ensemble。不过，论文最强的证据仍然是 **效率 + competitive accuracy**，不是“异构 HP 本身更优”。

对我们而言，它最值得继续做的是把这个机制从 tabular MLP 推到结构规则、算力利用率高的 backbone：ResNet/CNN 和固定图 GCN 都是很自然的验证对象。尤其固定单图多节点回归，graph topology 共享得比普通 mini-batch 场景更彻底，可能非常适合测试 packed GNN。

如果后续做研究，应该避免只复现“random HP pack > 一个 baseline”。更有价值的是证明：在同等 FLOPs / wall-clock / memory budget 下，packed architecture population 能否找到更互补的成员，并最终得到更强、更稳定的 ensemble。

## 值得继续追的问题

- ResNet / Conv 的 grouped packed kernels 在真实 GPU 上是否有接近 MLP packing 的吞吐收益；
- fixed-graph GCN 能否共享 sparse topology traversal，并把 member dimension 与 node dimension 高效融合；
- architecture heterogeneity 增大后，padding/mask 浪费从何时开始抵消 packing 收益；
- Muon 在不同 member width 下如何做严格的 masked matrix update；
- online selection 的 validation overfitting 能否用 nested validation、cross-fitting 或 training-loss proxy 缓解；
- candidate scoring 能否直接使用 marginal ensemble gain / error decorrelation，而不是只看 individual score；
- packed population 是否能在相同 wall-clock 下超过 tuned homogeneous ensemble，而不仅是达到相近精度；
- 是否存在一个通用 compiler / wrapper，自动把规则的 PyTorch module 转成 packed parameter + mask execution，而不需要每种 layer 手写实现。
