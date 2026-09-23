---
type: Research Paper Review
title: "Causilo Technical Report"
description: "Causilo 的 refine→revisit→compress 架构、分类/回归接口边界与性能证据；明确合成数据和预训练的公开缺口。"
resource: https://arxiv.org/abs/2609.22866
tags: [tabular-foundation-model, in-context-learning, synthetic-prior, efficient-attention, probabilistic-regression]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2609.22866
    title: "Causilo Technical Report"
  - id: code
    resource: https://github.com/nums-ai/causilo
    title: "nums-ai/causilo"
  - id: weights
    resource: https://huggingface.co/nums-ai/causilo
    title: "Causilo checkpoints"
---

# Causilo Technical Report

## Links

- [论文 arXiv:2609.22866v1](https://arxiv.org/abs/2609.22866) · [HTML 正文](https://arxiv.org/html/2609.22866) · [PDF](https://arxiv.org/pdf/2609.22866)
- [官方 GitHub](https://github.com/nums-ai/causilo) · [模型权重与许可证](https://huggingface.co/nums-ai/causilo)
- 代码实际阅读 commit：`4685c62b9445351786520e2e7b663ea302e996ab`；推理代码锁定的权重 revision：`94f2bd91db0737d4da59f347910662905ecb5a09`。以下代码事实仅对应此版本。

## 一句话结论

Causilo 是**表格分类/回归的预训练 ICL 模型**：`fit` 保存带标签的训练表作为 context，预测时不更新模型权重。它继承 TabICL 的 column→row→dataset-level ICL 路线，但在压缩 row embedding 前插入“列编码 → 行内细化 → 再次列交互”，让每个 feature-group token 先融合本行其它特征，再带着这层信息读取训练集。两个行模块都用固定数量的 summary tokens 做 cross-attention，避免特征维度上的平方注意力成本。（论文 §2.1–2.2；`execution/runner.py`、`nn/row.py`、`nn/column.py`）

**我们的判断：**这是以效果和效率展示为主的技术报告。最值得借鉴的是“延迟行压缩一轮，同时控制宽表推理成本”；论文没有隔离新模块、synthetic prior 和 8-view ensemble 的单项贡献，不能把全部 Elo 增益归因于架构。名称中的 *causilo* 不意味着模型在做因果识别；公开展示的是监督表格预测。（论文 §2–3）

## 文章摘要

模型在约 **3,600 万张合成表**上预训练，输入带标签 context 行与未标注 query 行。每三个相邻特征先聚合成一个 token，依次经过目标感知的列编码、行内细化、第二次列交互、行压缩，最后用 12 层 ICL Transformer 预测 query。分类和回归分别有约 **36.08M / 37.08M 参数**。评估覆盖 TabArena、BeyondArena、ScoringBench，重点展示准确率与推理延迟的 Pareto 关系。（论文摘要、§1–3）

## 方法拆解 / 这篇到底做了什么

设训练行数 `T`、查询行数 `Q`、保留特征数 `F`，`G=ceil(F/3)`。官方 checkpoint 中 `width=128`、列阶段各 3 层、行细化 3 层（2 轮更新加末次 broadcast）、行压缩 3 层、prediction 12 层；两项任务共用这一结构，分类/回归输出通道分别为 10/999。（HF checkpoint `config.json`；`model.py`、`nn/row.py`）

```text
(T+Q, F) → 3 特征一组 + sin/cos 数值编码 → (T+Q, G, 128)
  → column encoder: 128 inducing tokens 从训练行 gather，向所有行 broadcast
  → row refinement: 4 summary tokens 与组 token 交互，仍保留 (T+Q, G, 128)
  → second column block: 已融合同一行其他特征的 token 再读训练 context
  → row pool: 另 4 个 summary tokens 压缩每行为 512 维
  → 已标注行再次注入 y → 12 层 ICL attention → (Q, 10 / 999)
```

论文每个列阶段都只让**训练行**参与 gather；查询行读取聚合结果，不写入共享 context。预测阶段 context 行相互注意、query 行只读 context。因而多条 query 不应互相影响，这是避免 transductive query leakage 的关键。`ModelRunner.predict` 与 `PredictionBlock.forward` 确实按训练前缀建 K/V；缓存路径在 `ModelRunner.build_cache/predict_cached` 中复用同一语义。（论文 §2.1；`execution/runner.py`、`nn/prediction.py`）

这里的 `Engine` 不是网络 backbone：`model.py` 注册模块但没有 `forward()`；实际调用顺序由 `execution/runner.py` 决定。`engine.py` 负责加载权重、保存训练 context、构造 ensemble、选 direct/cached 路径和合并输出。数值使用 16 个可学习频率的 sin/cos 编码，缺失值有独立 embedding；分类 `y` 用 one-hot 投影，回归 `y` 先标准化再投影，且只注入训练行。（`nn/embeddings.py`、`data/dataset.py`）

**复杂度边界：**固定 4 个行 summary tokens 使行内 attention 对 feature-group 数近似线性；列模块也用 128 个固定 inducing tokens。全训练 context 的 self-attention 仍出现在最后的 ICL Transformer，所以 `N_context` 增大时并非整体线性。论文给出单次 forward 的注意力阶数为 `O((N_context + N_query)G + N_context(N_context + N_query))`，其中 `G≈F/3`。（论文 §2.2）

推理默认 `n_estimators=8`，成员共享权重，但轮换标准化、rank-to-Gaussian、robust、power 等特征变换，同时变换特征顺序和分类标签顺序。由于相邻三个特征被打包，特征置换也改变了分组。论文的主结果是**单一权重的多视图 ensemble**，并非一次模型 forward。（论文 §2.4；`data/ensemble.py`、`data/normalization.py`）

注意力层使用 RMSNorm、残差、SwiGLU；相对 TabICLv2 的 QASSMax，Causilo 将长度和 query 内容产生的缩放限制在正的有限范围，并加一项有界长度修正。这是独立于 `refine→revisit→compress` 的辅助修改，目前没有单项效果消融。（论文 §2.3；`nn/layers/attention.py`、`nn/layers/scaling.py`）

## 合成数据与预训练：当前能确认什么

论文只说明**完全使用约 3,600 万张合成表**，表的大小和 causalities 有变化；未给足生成器、任务混合比例、采样范围、训练课程等可复核细节。官方 GitHub 没有合成数据生成器或预训练管线。因此目前无法确认 synthetic prior 相对 TabICLv2 具体改变了什么，也无法把性能提升分解为架构、训练数据和训练预算的贡献。（论文摘要、§1；官方仓库文件树）

## 关键实验结果

- **TabArena（论文 2026-09-18 快照）**：51 个数据集，整体 **1785.4 Elo / 0.1043 s 每千测试行**。相对 TabICLv2，分类高约 **198 Elo**且快 **22.8%**；回归高约 **350 Elo**且快 **20.1%**。TabPFN-3.5、LimiX-2 的 Elo 更高，推理也更慢，因此 Causilo 主要是强准确率/低延迟折中。（论文 §3.1）
- **口径差异**：仓库 README 的 TabArena Full **1792.9 Elo** 排除了 system methods；论文含 system submissions，数字不可混为性能变化。仓库 H100 本地比较给出 **2.504 s fit / 0.251 s predict 每千行**，其中 `fit` 是上下文准备，不是梯度训练。（README、`docs/benchmarks/README.md`）
- **补充证据**：BeyondArena 142 个数据集整体 **1359 Elo**，但 baseline 只到 TabPFN-3；grouped regression 仍落后于一些传统模型。ScoringBench 在点预测与 CRPS 的性能/延迟图上进入 Pareto frontier。特征维度微基准拟合指数 `β=1.23`，但只测选定的行内算子，不能当作端到端复杂度。（论文 §2.2、§3.2–3.3）

## 关键分析

1. **主要架构贡献**是第二次列交互前保留 cell tokens，并通过 summary-mediated 行交互限制宽表成本。完整流水线比 TabICL 系更深，但最后仍把行压缩到固定维度，不能在 ICL 主干里继续逐特征更新。
2. **提升是完整系统的提升。**与 TabICLv2 的 Elo/速度比较同时包含架构、未知细节的 synthetic prior、bounded QASSMax 和默认 8-view ensemble。论文缺少同等先验/预算下的模块消融，尚不能单独归因。
3. **速度结论要分口径。**TabArena 图是公开榜单统计的 predict 延迟；仓库单机 H100 复测给出更高的 0.251 s/1k。两组都能支持“推理快”的方向，但数值和 baseline 池不可混用；8-view ensemble、训练行数量、是否 K/V cache 都影响真实部署耗时。

## Insights

- `refine → revisit context → compress` 是可复用的设计：如果一开始按列读取上下文，先做一轮行内融合再按列读第二遍，才有机会让跨特征关系影响跨样本检索。
- 固定 latent 数使**特征维度**可扩展，但 `context` 的二次方项还在 ICL 主干。宽表与大样本是不同扩展问题，不能用一条复杂度曲线概括。
- 先读真实 `fit` 语义：表格 foundation model 的 scikit-learn `fit` 常常只是构建 context/cache；评估 fit 成本时仍应计入这部分计算与内存。

## 我们的观点

值得继续关注，尤其适合作为**宽表、低延迟 ICL**的架构候选。它的性能证据跨三个基准，且推理实现开源，可在本地直接跑；但机制解释还欠同预算 ablation。默认 8-view 已是小型推理 ensemble，评估单视图与各阶段贡献会比继续比较总榜 Elo 更有研究价值。模型权重采用单独的 Causilo License v1.0：非商业研究可用，商业/生产以及 hosted/API/SaaS 场景需要另行授权；代码本身为 Apache-2.0。（论文 §4；官方 README）

## GitHub / Code Analysis

- **接口支持矩阵**（`estimators.py`、`data/dataset.py`、`engine.py`，以上述 commit 为准）：

  | 能力 | 实际行为 |
  | --- | --- |
  | 分类 | 单目标 binary/multiclass；`predict_proba` 返回每类概率；原生 head 为 10 类，`>10` 用 ECOC 多次推理和解码 |
  | 回归 | 单个连续目标；默认点预测是 999 个输出通道排序后的均值，还可返回 median、指定分位数或 999 个 native quantiles |
  | 多目标 / 多标签 | 不原生支持；`y` 为 `(n,1)` 时压平，`(n,m>1)` 会在 `PreparedDataset.prepare` 拒绝 |
  | `sample_weight` / `class_weight` | 不支持；`fit(X, y)` 和构造函数均没有相应参数，`fit` 也不优化权重 |
  | 缺失值 | 特征缺失值可用、未见类别按缺失处理；目标缺失值拒绝，datetime/timedelta 特征需先编码 |

- `src/causilo/estimators.py` 暴露 sklearn 分类/回归接口；`engine.py` 负责 `fit` 状态、8-view ensemble、ECOC 和结果合并；`data/encoding.py` 负责分类列识别、类别编码、常量列删除和 query schema 校验。时间/日期列要求用户先编码，未见 time-aware cutoff 语义。
- `model.py` 只注册 checkpoint 定义的模块；`execution/runner.py` 决定实际执行顺序；`nn/column.py`、`nn/row.py`、`nn/prediction.py` 分别实现两次列交互、行细化/压缩、最后的 context-only ICL。这些源文件支持论文的架构叙述。
- `checkpoints.py` 锁定 HF revision，并验证配置哈希与 safetensors metadata。`use_kv_cache=True` 时 `fit` 预计算训练 K/V，方便重复预测；默认关闭。`execution/direct.py`、`execution/recompute.py` 和 `execution/memory.py` 有分批与内存不足时重计算路径，但没有改变模型所用训练 context 的语义。
- 公开仓库可复核**推理机制和部分评测产物**；未发现合成数据生成、预训练与架构消融的完整代码。`docs/benchmarks/provenance.json` 记录了 TabArena 上游 commit、软件版本、816 个 split 和文件哈希，但公开文件本身不足以独立重训 36M synthetic prior。

## 值得继续追的问题

1. 同样 synthetic prior、训练预算与 8-view ensemble 下，移除第二次列交互或 row refinement，各自造成多少 Elo/延迟变化？
2. 不同训练行数、特征数与 `n_estimators=1/4/8` 下，准确率与真实 GPU 峰值如何变化？长 context 的 `N_context²` 何时成为瓶颈？
3. grouped/temporal split 下相对 RealMLP、CatBoost 的劣势来自合成 prior、context 选择还是分布漂移？
4. 999 通道排序成分位数后的概率校准、尾部外推与 CRPS 是否在外部回归任务上稳定？
