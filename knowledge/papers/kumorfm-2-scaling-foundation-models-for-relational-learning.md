---
type: Research Paper Review
title: "KumoRFM-2: Scaling Foundation Models for Relational Learning"
description: "KumoRFM-2 的关系子图 ICL、相对 TabICLv2 的算法边界、context/subgraph scaling 与 PQL 系统设计；对照公开 adapted implementation 和商用许可证。"
resource: https://arxiv.org/abs/2604.12596
tags: [relational-foundation-model, in-context-learning, tabicl, predictive-query-language, temporal-consistency, scaling]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2604.12596v1
    title: "KumoRFM-2 technical report"
  - id: code
    resource: https://github.com/NVIDIA/structured-data-models
    title: "Official KumoRelational adapted implementation"
  - id: weights
    resource: https://huggingface.co/nvidia/Kumo-Relational
    title: "Kumo-Relational checkpoints and OpenMDW 1.1 license"
  - id: pql
    resource: https://arxiv.org/abs/2602.09572v3
    title: "Predictive Query Language specification (supplementary source)"
  - id: tabicl
    resource: https://arxiv.org/abs/2602.11139
    title: "TabICLv2 architecture comparison"
---

# KumoRFM-2: Scaling Foundation Models for Relational Learning

## Links

- [arXiv:2604.12596v1](https://arxiv.org/abs/2604.12596v1) · [HTML](https://arxiv.org/html/2604.12596v1) · [PDF](https://arxiv.org/pdf/2604.12596)
- [官方项目](https://kumo.ai) · [Kumo 文档](https://kumo.ai/docs) · [原始 examples / benchmarks 仓库](https://github.com/kumo-ai/kumo-rfm)
- [当前官方本地推理代码：NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) · [公开权重](https://huggingface.co/nvidia/Kumo-Relational)
- [PQL 独立论文](https://arxiv.org/abs/2602.09572v3)：只作为本篇系统分析的补充来源，不另建论文条目。
- [本仓库 TabICLv2 笔记](tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model.md) · [RelArena / TabPFN-Rel 对照](advancing-open-and-reproducible-relational-learning-relarena-tabpfn-rel-rpi.md)

讨论与代码核查日期：**2026-10-12—2026-10-13**。论文作者为 Valter Hudovernik、Federico López、Vid Kocijan、Akihiro Nitta、Jan Eric Lenssen、Jure Leskovec、Matthias Fey；v1 发布于 2026-04-14。

实际阅读代码 commit：`5e5284a511f42b85873415a0a3099a9902c7a03b`。`model.py` 指定 checkpoint tag `v1.0.1`，核查对应 HF revision 为 `2bd603d3d8f25f67a7aaa8579908595e20567e22`。下文代码事实仅对应此快照；源码通过临时 clone 阅读，未 vendoring，也未运行 GPU benchmark。

## 一句话结论

KumoRFM-2 把一个 ICL 样本从单行特征扩展为**实体在某个 anchor time 的关系子图**，在模型内部学习表内特征组合与跨表聚合，并用 PQL 自动定义目标、生成历史标签和取回 context。论文展示了强的关系学习结果；相对 TabICLv2 最值得保留的是关系信息的可学习利用与任务构造，而非 early label injection 或分层 row compression 本身。

**我们的判断：**这是一篇算法、数据配方与系统工程共同推进的技术报告。PQL 将 prediction grain、label generation 和时间窗口提升为可执行任务规范，很有长期价值。当前已有允许商用的官方权重与本地代码，但公开 `KumoRelational` 明确是 **adapted version**，不能把它等同于论文全部架构、托管模型成绩或 500B+ 行生产系统。

## 文章摘要

问题是在包含多张关联表的数据库上，直接预测用户流失、消费额、业务字段等目标，减少手工 flattening、标签构造和任务训练。数据库记录对应节点，PK/FK 对应边；每个学习样本由实体、anchor time 和标签定义。模型先进行任务条件化的表内编码，再利用关系结构与跨样本交互完成 ICL，也支持任务微调。（论文 §2–3）

预训练混合 SCM 生成的合成表/图与公开真实关系数据库中的预测查询，并从单表逐步过渡到更复杂关系任务。评测覆盖 15 个数据库、41 个任务，包括 RelBenchV1/V2、SALT、4DBInfer。主要收益出现在关系上下文、多类别与回归；论文同时展示 context、子图深度/宽度和缺失/噪声实验，但不足以隔离架构、训练数据与采样策略的单项贡献。（论文 §3–4）

## 方法拆解 / 这篇到底做了什么

### Prediction grain 与时间语义

一个样本是 `(entity v_i, anchor time t_i, target y_i)`，输入是 `G_{≤t_i}[v_i]`。ICL 输入为一组带标签历史子图与目标实体的未标注子图：

```text
context = {(历史实体子图, 历史标签)}
query   = 目标实体在预测时间可见的子图
冻结 RFM 参数 → 预测 query 的标签
```

这里的 few-shot / 无任务训练并不表示没有目标标签：context 仍包含监督信息。未来聚合任务的标签来自 anchor time 之后的窗口，但特征来自此前；context 标签窗口还必须在预测时已经完整可观测。时间列语义、历史字段修订和标签成熟时间都影响 point-in-time correctness，不能只靠一个 `≤t` 过滤条件解决。（论文 §2；后半段是我们的工程判断）

### 论文架构

```text
数据库 schema + context task table
  → 历史实体子图
  → 轻量表内 column/row attention，早期注入 context targets
  → row embeddings
  → 较大的 foreign-key / graph attention 与 cross-sample attention 网络
  → 目标实体 readout
```

论文 Fig.3 / §3 的重点是四个交互轴：表内行、列，以及数据库级外键、跨样本。先压缩 row representation 再跨表计算，避免所有 cells 的全局平方 attention。论文没有给足层数、精确 mask 与所有交替顺序；不可据图补全具体实现。

### 相对 TabICLv2：新增在哪，哪些不是新增

| 比较点 | TabICLv2 | KumoRFM-2 的扩展与边界 |
| --- | --- | --- |
| Context 单元 | `(x_i, y_i)` 单行样本 | `(G_{≤t_i}[v_i], y_i)`，保留相关记录与关系 |
| 表内编码 | column embedding → row interaction → ICL | 论文描述表内编码后加入关系推理；公开版本直接复用 TabICLv2 组件 |
| Early target conditioning | 已有 target-aware embedding | 扩展到相关表记录，不是相对 TabICLv2 首创 |
| 分层压缩 | 已先压缩 row embedding 再做 dataset-level ICL | 增加关系结构处理层，不是首次采用分层架构 |
| 多表聚合 | 通常需外部 join / feature construction | 模型内部学习记录级表示及关系消息聚合 |
| 时间信息与 context selection | 可由用户提供 lag features / 历史样本 | 论文组合实体自身历史、lagged targets 与近期全局 context |
| 长 context attention | QASSMax 有具体公式与消融 | 不能由 Kumo 的系统容量宣称推出更强长 context；公开适配版本也复用 QASSMax |

来源：TabICLv2 §3、Kumo §3 与下文源码。公开模型卡还明确说明 Kumo Relational **fine-tunes pretrained TabICLv2 weights**；这比“结构类似”更直接，但不能因此反推原始托管模型的所有初始化细节。

### Fig.4 的 expressivity 示例：信息损失发生在哪

目标是判断某实体是否有至少一条子表记录同时满足 `A=1 AND B=1`：

```text
实体甲的子表行：(1,0), (0,1) → y=0
实体乙的子表行：(1,1), (0,0) → y=1
```

分别对 A、B 做 sum/mean/max 会得到一样的边际统计，丢失同一行的配对关系。论文在该构造中报告固定逐列 encoder 的 AUROC=0.5，Kumo=1.0。（论文 §3 / Fig.4）

**我们的分析：**这是选定 flattening pipeline 的信息损失，不是 TabICLv2 学不会 AND，也不是所有 DFS 配方都无法表达。加入 `sum(A*B)` 就能恢复信号。Kumo 的价值是先学习行内组合再聚合关系记录，从而减少人工指定 cross-features；不能把 toy example 当成对所有单表模型的普遍不可能性证明。

## Scaling 工作：扩了什么

| 方向 | 工作 / 实验 | 解释条件 |
| --- | --- | --- |
| 预训练分布 | 扩展合成表/图与真实数据库查询；单表→关系任务的阶段训练 | 没有充分披露数据量、算力和参数量 scaling curves |
| Context 条数 | SALT subset，Fig.6(a)，少量 context 已有效，增加后持续改善 | 与 XGBoost / TabICLv2 的比较不能隔离训练配方差异 |
| 子图深度 | SALT subset，0–6 hops；0-hop 接近 TabICLv2，1-hop 明显提升，约 4-hop 饱和 | 是关系信息范围的扩展，不是 Transformer 层数 scaling |
| 子图宽度 | RelBenchV1 分类 subset，固定 2 hops、每跳 1–64 邻居 | 平均改善，但 `f1 driver-dnf` 约 n=4 达峰后下降，`amazon item-churn` 持续受益 |
| 任务数据利用 | SALT fine-tuning 使用全部训练标签，ICL coverage 约 0.5% | 改变了适配方式与标签预算，不是纯 ICL scaling |
| 数据库与请求容量 | SQL pushdown、SSD graph engine、无状态多租户服务 | 500B+ 行是系统访问容量，不是单次模型 context 或预训练行数 |

论文 §3、§4.4 / Fig.6、§4.5、§2。宽度与深度共同决定每个样本的图大小；我们的判断是，关系 ICL 的预算应同时记录 context 条数、节点数、边数和 cells，而不能只报“10k examples”。论文没有建立参数量×数据量×算力的 scaling law。

## 系统工程与 PQL 讨论

### 论文系统的执行路径

```text
业务问题 / agent → PQL + schema + anchor time
  → 定义目标、实体、时间窗口
  → 生成已观测历史标签 + 截断历史子图
  → 内存 / 数据库 SQL pushdown / SSD graph engine
  → context 请求 → RFM inference → 预测 / embeddings / evaluation / explanations
```

论文 §2 描述 schema-level 接入、手动覆盖主外键/semantic types、views / derived columns；数据库 connector 沿 metapaths 用递归 SQL 构造 context，无需物化中间图；另一条 graph engine 路径使用 memory-mapped 数据结构，作者报告 500B+ 行、最高 5GB/s 与 20M lookups/s。无状态 serving 不绑定数据库或租户，每次请求携带 context，便于横向复制。以上是**论文系统说明**，不是本次源码核查或独立压测结论。

SDK 提供 context budget、子图邻居数、ensemble 等控制，并提供 agent skills / MCP 接口。早期论文的 hosted inference 和 enterprise fine-tuning，与当前 SDM 本地模型接口应分开理解；SDM `fit` 保留 context cache，也不等同于无状态托管服务的内部实现。（论文 §5；源码 `models/base.py`）

### PQL：把任务变成可执行规范

```sql
PREDICT SUM(orders.price, 0, 30, days)
FOR EACH users.user_id
```

这定义的是每个用户未来 30 天的消费额：预测粒度为 user，target 为未来窗口的数值聚合。对历史 anchor t，系统可以生成 `(user_id, t, sum(price in (t,t+30d]))`，并取 t 时可见的特征。预测时未来 target 未知，由模型估计；PQL 引擎本身不“查询出未来”。SUM 的窗口边界为起点不含、终点包含。（[官方 SUM 文档](https://kumo.ai/docs/reference/sum/)）

补充阅读 PQL 独立论文 v3 §3.9：ANTLR4 解析→AST/schema validation→逻辑计划→后端优化与执行；包含 schema-based join inference、实体有效时间检查等。它既能为 supervised RDL 构造 training table，也能为 RFM 构造 ICL context，不绑定某一个 foundation model。独立论文的 Spark batch 路径与低延迟实现，不能自动当作 SDM 内置功能。

**我们的观点：**PQL 的前瞻性在于把 prediction grain、label generation、时间窗口和实体集合提升到语言层。自然语言 agent 可以产生可审核任务规范，而不必每次临时编写整个数据管线；模型和数据后端以后可以替换，任务定义仍可复用。它补上的是 feature engineering 和 model selection 之外常被忽略的 supervision construction。

**边界：**schema 的 PK/FK、时间列与 semantic types 必须正确；事件时间与数据可用时间不同、字段被事后修订时，还需要历史版本或双时间语义。PQL 能规范构造流程，不会自动解决所有数据质量、预测校准或因果识别问题。独立论文中的 `ASSUMING` 也不能仅凭语法推出干预效应可识别。（数据库假设见 PQL §3.2；其余为我们的分析）

## 关键实验结果

### 协议与 baseline fairness

论文 §4 声称四套 benchmark 未用于预训练；这是作者陈述，不是独立污染审计。主成绩使用冻结基础模型 ICL，**context 取自 train + validation（若存在）**，最多 10k examples；不应只写“官方 split，所以标签预算完全一致”。分类关系模型使用 2-hop 子图。Baselines 尽量沿用已发表数字，自跑项用星号标记。（§4、§4.1）

裸单表 baseline 通常只看 entity table；DS+ 表示手工特征，DFS+ 表示自动 flattening。比较必须分清信息预算与预处理：不能用裸单表模型的低分证明 backbone 不如关系模型。评测包含 12+9 个 RelBenchV1 任务、5+2 个 V2 任务、8 个 SALT 任务、5 个 4DBInfer 任务，共 41 个。（Tables 1–2）

| Benchmark / 指标 | KumoRFM-2 | 主要比较 | 来源 |
| --- | --- | --- | --- |
| RelBenchV1 分类，平均 AUROC ↑ | 79.60；rank 3.08 | v1 76.71、RelGNN 78.06、DS+LightGBM 75.62 | Table 3 |
| RelBenchV1 回归，normalized MAE ↓ | 0.822；rank 2.67 | v1 0.908、RelGNN 0.861、DS+LightGBM 0.879 | Table 7 |
| RelBenchV2 分类，平均 AUROC ↑ | 79.96；rank 1.75 | GraphSAGE 78.19、v1 76.22、LightGBM 71.79 | Table 4 |
| RelBenchV2 回归，normalized MAE ↓ | 0.601；rank 1.0 | GraphSAGE 0.626、v1 0.712 | Table 8 |
| SALT，平均 MRR ↑ | ICL 0.83；FT 0.89 | v1 0.80、DS+CARTE 0.79、DS+AutoGluon 0.77、DS+TabICLv2 0.71 | Tables 6、9 |
| 4DBInfer，平均 AUROC ↑ | 79.96；rank 3.6 | HGT 79.94；rank 3.4，v1 79.17 | Table 5 |

Normalized MAE 是各任务相对 LightGBM 的误差归一化后平均，不能按原始 MAE 混合不同单位。RelBenchV1 的平均优势不等于所有任务胜出；大训练集的 amazon / stack / hm 分类任务仍有监督模型优势。（§4.1、§4.3）

**讨论中的数字纠正：**RelBenchV2 的 `80.65 / 80.51` 分别是 Kumo / GraphSAGE 的 `ratebeer dormant` 单任务分数，均值是 `79.96 / 78.19`，差 **1.77 AUROC points**。此前“二者均值只差 0.02”错误；真正基本持平的是 4DBInfer 的 `79.96 / 79.94`。此处按 Table 4 列头核对，不沿用早期对话的错位读取。

SALT 的 Sales Group（589 类）从 v1 MRR 0.38 到 v2 0.61，FT 后 0.74；Payment Terms 从 ICL 0.68 到 FT 0.86。作者报告 FT 每任务少于 2 分钟，但这不是我们复测的 wall-clock 结果，也不是公开 checkpoint 已复现所有这些成绩。（Tables 6、9）

### 鲁棒性证据

Fig.7 在 RelBenchV1 分类 subset 上改变 missing-value 比例、FK 边保留情况与噪声列。模型在最多 75% 边被删除时仍保持较强效果；关系完全移除时退化到单表模型水平。噪声列下 v2 比 v1 稳定。特征稀缺实验是**随机引入缺失值**，不能误记成直接删除整列。作者将抗噪声归因于早期 task conditioning，但它不是仅替换该模块、保持其余训练相同的严格因果消融。（§4.4）

## GitHub / Code Analysis

### 公开范围与三个不同对象

1. `kumo-ai/kumo-rfm`：论文引用的 examples / benchmarks / issue tracking 入口。
2. `NVIDIA/structured-data-models` 中的 `KumoRelational`：本次实际阅读的可本地执行 **adapted implementation**。
3. 论文描述的 Kumo 托管/企业系统：PQL、SQL pushdown、SSD engine、serving 与 fine-tuning 产品能力。

当前公开权重的存在，纠正了早期讨论“只有 API、无公开权重”的判断。另一方面，代码 docstring 明确写 adapted version / high-level architecture；没有依据把三个对象当成完全一致的发布物。

### 入口、模块与配置

源码前缀为 `sdm/models/kumo/relational/`。以下链接固定到实际阅读 commit：

- [model.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/models/kumo/relational/model.py)：`KumoRelational` 加载 classifier/regressor；`_KumoRelational.forward` 执行实际计算。
- [task.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/models/kumo/relational/task.py)：task→entity join、邻域 task assignment。
- [graph.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/models/kumo/relational/graph.py)：各表映射为 homogeneous graph，join 生成双向边及不同方向 edge types。
- [invariant_gnn.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/models/kumo/relational/invariant_gnn.py)：schema-agnostic message passing。
- [models/base.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/models/base.py)：统一 preprocessing、ensemble、`fit/predict` context cache。

初始化分别为 classification / regression 加载 `classifier.pt` / `regressor.pt`。默认 feature types 声明为 numerical / datetime，target 为 categorical / numerical；**这不等于原始数据库类别特征完全不能使用**：继承的 TabICLv2 recipe 先对齐类别再转 numerical。数值路径包含均值填补、删除常量列、标准化、clipping、Identity/PowerTransform 轮换与列置换；分类目标对齐/置换，回归目标标准化。输出排序回归分位数、平均 estimator，并对分类执行 softmax。Kumo recipe 另外对 related tables 添加 minute/hour/weekday/day/month calendar features。（`tabiclv2/recipe.py`、`kumo/relational/recipe.py`）

默认核心配置：128 channels、3 层 column encoder、3 层 row encoder、128 inducing points、4 readout tokens，因此 row embedding 为 512 维；之后接 GNN 和 12 层 ICL block，分类基础 head 10 类、回归 999 分位数。多类别分别在前端使用 mixed-radix label embedding，在 ICL block 使用 hierarchical classification。`max_train_size=20_000` 用于列阶段 key 数量上限，超过后随机选 keys；不能直接把它解释成最终 ICL context 的硬上限。

### 真实 forward 路径与论文差异

```text
RelatedTables + task/context/query tables
  → TaskGraph：PK/FK join，定位 readout，传播 task-row assignment
  → 每表 numerical features + task features + relative-time features
  → 共享 TabICLv2 RowEmbedding（早期标签、column attention、row attention）
  → context/query 各自执行 InvariantGNN
  → entity readout → TabICLv2 ICLBlock → 分类 logits / 回归 quantiles
```

`_embed_table` 将 task-row assignment 对应的 `y` 分发给相关表记录，未关联行不作为 labeled keys。列 attention 的 K/V 来自 context，query 行不写入共享 context；最后的 ICL block 同样以 context prefix 为 K/V，context 在此再次注入标签。（`model.py`、`tabiclv2/row_embedding.py`、`tabiclv2/icl.py`）

**重要差异：公开跨表模块不是 graph attention。**`InvariantGNN` 每跳先线性变换源节点，加 edge-type embedding，再调用 `segment_multi_reduce` 计算 **sum / mean / std / min / max**；拼接统计后用线性映射、skip、LayerNorm/GELU 更新节点。没有邻接边上的 QK softmax 权重。edge-type embeddings 由随机向量经线性层生成，context/query 通过 cache 复用。论文的 foreign-key attention 描述不能原样写成这份源码的事实。

公开表内模块也不是反复 column↔row 交替：继承的 `RowEmbedding` 先完成 column layers，再完成 row layers。QASSMax 在 column inducing gather 与 ICL block 中启用。因此相对 TabICLv2 的公开实现可概括为“复用任务感知 row encoder 和 ICL backbone，中间加入关系消息聚合”，而不是另起一套全部未知的注意力架构。

### 共享记录与时间子图：讨论问题的源码答案

`TaskGraph.from_input` 明确假设不同样本邻域不重叠，且每个 task row 对应不同的 entity-table row。直接提供共享图不自动解决多样本标签条件化。

配套 [relational/sampler.py](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/sdm/relational/sampler.py) 会添加 example ID，并把它纳入 relationship / task-link composite keys，使不同样本中的同一实体记录保持 disjoint linkage。PyG backend 使用 `disjoint=True`；时间采样传入 seed times 与各表 node times。这解释了如何避免一条子表记录同时被多个目标标签直接条件化：按样本构造分离的记录实例，而非在单一共享节点上混合标签。

`_get_rel_time` 另计算 seed timestamp 减 record timestamp 的天数，执行 signed-log1p 与标准化。这是时间**特征编码**，并不负责从原始表中删除未来记录；cutoff 必须由采样层配置，不能仅因模型收了时间列就宣称无泄漏。

### fit / predict、evaluation 与复现条件

`ICLModel.fit` 在 no-grad 路径预处理 context 并记录/freeze cache；不执行任务梯度训练。`predict` 回放 preprocessing 和各层缓存。它是有状态的本地 context caching 接口，不是论文所说的 enterprise fine-tuning。公开本地 forward 加载 HF 权重后无需托管预测 API。

[RelBench example](https://github.com/NVIDIA/structured-data-models/blob/5e5284a511f42b85873415a0a3099a9902c7a03b/examples/relational/relbench/main.py) 拼接 train+validation 为 context，按预算选取样本，可按 estimator 选不同 context；通过带时间列的 sampler 构造图，调用 `fit`，再批量 `predict`。回归使用 `q500` 作为中位数点预测。仓库还提供 SALT / RelArena examples 与 tests；本次仅阅读，未下载 checkpoint 或运行其成绩。

依赖：Python≥3.11、PyTorch≥2.7、numpy、pyarrow、huggingface_hub 等。关系采样根据数据设备选择 cuGraph / pyg-lib backend，不能把纯 PyTorch forward 的可运行性等同于全部 sampler 在任意平台开箱即用。论文的 Python≥3.10 SDK 是另一条发布路径。

本次模型调用链使用显式 `TableTensor / RelatedTables`；未据此确认完整 PQL 编译器、Snowflake pushdown 或 500B SSD engine 是 SDM 的组成部分。公开 `v1.0.1` checkpoint 与论文 benchmark 的逐项对应关系也尚未验证。

### License 与商用

已读取 HF revision `2bd603d3d8f25f67a7aaa8579908595e20567e22` 的 [LICENSE](https://huggingface.co/nvidia/Kumo-Relational/blob/2bd603d3d8f25f67a7aaa8579908595e20567e22/LICENSE)：**OpenMDW 1.1** 免费授权使用、修改与分发 Model Materials，允许商用；再分发需保留协议、版权与来源声明，对输出不附加使用/修改/分享限制。协议含特定专利/版权诉讼触发的权利终止条款，不应只记成“无限制”。

SDM 的 NVIDIA-authored code 使用 Apache-2.0，复用 TabICLv2 组件保留 BSD-3-Clause；模型卡注明权重 fine-tune 自 TabICLv2，附 `THIRD-PARTY-NOTICES.txt`。本地商用需要遵守各发布物的对应许可。API 服务与 NIM container 另有条款，不能拿示例仓库 MIT 或公开权重 OpenMDW 代替。（官方代码 LICENSE / NOTICE、模型卡与权重 LICENSE）

## 关键分析

- **算法贡献与产品能力不能混算。**关系内特征学习和消息交互属于模型；context selection、标签生成、数据下推和 serving 属于配方/系统。准确率提升无法只归因于 early conditioning。
- **Scaling 的证据以输入信息量为主。**Fig.6 说明更多样本与更丰富关系上下文有用；没有充分披露算力/参数 scaling law。500B 行宣称说明数据访问后端，不能写成模型看过或同时 attention 了 500B 行。
- **公开实现提高了可用性，但没有消除论文复现缺口。**可以本地研究 row encoder→GNN→ICL，也允许商用；还需确认 adapted checkpoint、预训练配方与论文结果的对应关系。
- **平均榜单与公平比较需要拆开。**V2 分类确有 1.77 points 均值优势；4DBInfer 的均值接近且 HGT 平均 rank 更好。train+validation context、baseline 信息范围、历史发表结果与 API/本地发布差异都应一起记录。

## Insights

1. **自动建模仍需要自动构造监督。**强 backbone 不会自行知道 prediction grain 和 label horizon；PQL 将这一层变成可复核 contract。
2. **关系学习优势常来自保留信息。**固定逐列聚合丢失行内配对之后，下游强模型无法恢复；但补充正确 cross-features 可以缩小差距，不能把 pipeline 损失错归因于 backbone。
3. **关系 ICL 有两个预算轴。**context 数量与每个 context 的子图信息量独立变化，应联合研究收益、延迟和显存。
4. **Temporal sampling 与 temporal encoding 是不同职责。**相对时间 embedding 提供模型信号，采样 cutoff/历史版本才控制数据可用性。
5. **复用成熟 tabular backbone 是自然的关系模型路线。**公开 adapted version 提示，可先验证一个浅的 learned relational adapter 是否已经足够，再考虑更复杂 FK attention；这仍需公平 ablation。
6. **许可证、权重与产品系统是不同资产。**开放可商用 checkpoint 不自动开放全部数据引擎和服务实现，也不证明复现了托管模型成绩。

## 我们的观点

我们给它较高阅读优先级，尤其关注 relational foundation model、数据库原生预测与 agent 调用预测工具。论文的工程贡献很重，但系统抽象有实质价值；PQL 比单纯的 MCP wrapper 更值得长期追踪，因为它定义了任务与标签语义，而不是只换了一种调用方式。

算法上最值得借鉴的是在压缩与聚合过程中保留任务相关的记录级关系信号。早期标签注入和分层 row embedding 已见于 TabICLv2；当前公开版本还直接继承其权重和模块，所以不应把所有升级写成全新算法。若目标是私有数据本地研究与商用，公开 OpenMDW 权重使可行性明显高于我们初读时的 API-only 印象，但复现论文仍有待验证。

## 值得继续追的问题

- 相同 TabICLv2 初始化、训练数据与预算下，GNN、FK attention、context selection 分别贡献多少？
- 公开 adapted model 对应论文哪个模型版本，能否以相同 10k context / 2-hop 配方复现 Tables 3–9？
- 在匹配信息预算的 DS/DFS+TabICLv2 baseline 下，关系模型还剩多少优势？加入行内 cross-features 是否改变 Fig.4 的结论？
- 在固定总节点/cell 预算下，增加 context 条数与增加每样本子图宽度哪个更有效？
- Disjoint sampling 的重复记录成本多大，能否共享 raw storage 而保持样本级 task conditioning 和独立 cutoff？
- 真实数据库中的 late arrival、字段修订、label maturation 与 bitemporal history 如何进入 PQL 计划？
- PQL task contract 如何版本化，哪些 schema/业务定义变化需要重建 context、重新评估或微调？
- 公开本地概率/分位数预测的校准、成本和 throughput 怎样，是否能达到托管服务的延迟与完整功能范围？
