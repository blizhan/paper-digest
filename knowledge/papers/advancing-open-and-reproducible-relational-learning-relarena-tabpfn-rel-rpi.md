---
type: Research Paper Review
title: "Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI"
description: "Relational-learning benchmark 与工程体系分析，重点讨论 DFS flattening + TabPFN、prediction grain、temporal protocol 与公开实现。"
resource: https://arxiv.org/abs/2608.16319
tags: [relational-learning, tabular-foundation-model, benchmark, reproducibility]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2608.16319
    title: "Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI"
  - id: code
    resource: https://github.com/PriorLabs/relarena
    title: PriorLabs/relarena
---

# Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI

## Links

- Paper: [arXiv:2608.16319](https://arxiv.org/abs/2608.16319)
- PDF: [arXiv PDF](https://arxiv.org/pdf/2608.16319)
- Hugging Face Papers: [2608.16319](https://huggingface.co/papers/2608.16319)
- Official code: [PriorLabs/relarena](https://github.com/PriorLabs/relarena)
- 本笔记代码分析基于官方仓库 commit `e89002200e18be6d8d7a55f8a5ab50c993ce4d5d`。

## 一句话结论

这篇与其说是在提出一个全新的 relational foundation model，不如说是在同时做三件事：建立更统一、可复现的 relational benchmark；把成熟的 DFS relational featurization 接到 TabPFN-3 上形成一个很强的 baseline；再提供一个把真实关系库预测任务描述成统一接口的 RPI。

真正值得关注的实验信号是：**在 entity-level relational forecasting 上，关系库 flattening + 强 tabular model 仍然是必须认真对待的路线，复杂 GNN / relational Transformer 并没有形成稳定碾压。**

## 文章摘要

论文认为 relational learning 目前的一个核心问题不是“完全没有模型”，而是不同工作之间的数据加载、时间切分、调参协议、任务定义和评估方式不够统一，导致结果难以直接比较和复现。

作者因此开源了三个相互配套的组件：

1. **RelArena-α**：面向 RelBench v1 的统一评测框架，标准化数据加载、validation/test protocol、hyperparameter tuning、结果聚合和系统级 submission。
2. **TabPFN-Rel**：把 relational database 通过 Deep Feature Synthesis（DFS）转成 tabular features，再使用 TabPFN-3 做预测的 relational harness。
3. **RPI（Relational Predictive Interface）**：用 declarative YAML 来描述数据库 schema、时间切分和预测任务，让 RelArena 中注册的模型可以应用到新的关系数据库，而不需要每个任务重新写一套 Python task generator。

RelArena-α 当前仍是面向 researcher / early adopter 的 α-release，主要覆盖 RelBench v1 的 **entity-level forecasting**，并不是通用 production relational ML platform。

## 这篇到底做了什么

### 1. RelArena-α：先把比赛规则统一

这一部分其实是整篇最重要的工程贡献。它统一了：

- 数据和任务加载；
- model API；
- train / validation / test 的 temporal protocol；
- tuning regime；
- model 与 system submission 的边界；
- rank、win rate、normalized score、bootstrapped Elo 等评估；
- preprocessing cache 与结果缓存。

尤其关键的是 temporal tuning。官方说明 inner split 使用 validation cutoff 冻结数据库状态，outer split 使用 test cutoff，避免 tuning 阶段和最终评估阶段看到不一致的数据库历史。这看起来是 protocol 细节，但对于关系型时间任务很容易直接改变结果，也很容易产生隐蔽 leakage。

### 2. TabPFN-Rel：DFS flattening + TabPFN-3

核心流程可以压缩成：

```text
relational DB
    ↓
沿 PK/FK schema 的 join paths 做 DFS aggregation
    ↓
一张 target-grain 的 flat feature table
    ↓
附加可选 calendar / history lag / raw text features
    ↓
选择 in-context rows
    ↓
TabPFN-3 local 或 hosted API
```

因此 TabPFN-Rel 的“Rel”主要发生在 **feature construction + temporal/context handling**，而不是一个新的 GNN message-passing architecture。

### 3. RPI：把“什么叫一个 relational prediction task”变成接口

RPI 用 YAML 声明数据库和预测任务，再通过统一的 `PredictiveQuery` 接口调用 RelArena 中注册的方法。这个方向很有价值，因为真实 relational ML 的困难经常不在 estimator 本身，而在：

- target entity 是什么；
- cutoff time 怎么定义；
- 哪些历史记录允许被看到；
- label 怎么生成；
- train / validation / test 的数据库状态分别是什么。

不过当前 RPI 仍然有明确边界：它主要服务于 entity-level forecasting，并且官方也承认 task specification 仍然是开放研究问题，现阶段 safeguards 有限。

## TabPFN-Rel 方法拆解

### 不是简单直接 JOIN

如果有：

```text
users
  user_id

orders
  order_id
  user_id
  amount

order_items
  order_id
  product_id
  quantity
```

最朴素的做法是：

```sql
users
JOIN orders
JOIN order_items
```

这样一个 user 会因为多个 orders / items 重复成很多行，最终样本粒度变成 `user-order-item`，而不是原来的 prediction target。

TabPFN-Rel 走的是 DFS：沿 schema 的主外键关系遍历 join paths，并把 one-to-many 等关系通过 aggregation 压回预测对象的粒度。例如可能得到：

```text
user_id
num_orders
sum_order_amount
avg_order_amount
max_order_amount
num_products_bought
avg_quantity
...
```

所以更接近：

> 自动 SQL JOIN + 自动 GROUP BY / aggregation feature engineering + TabPFN。

官方实现会把 DFS 最大深度 `d` 在 `{2, 3, 4}` 上做 task-level tuning。

### 预测单位 / grain

这里需要比“一个 user 一行”再精确一点。

对于 temporal entity forecasting，真正的 prediction unit 往往是：

```text
(entity, cutoff_time)
```

例如预测 user 在某个 cutoff 之后是否 churn，那么一行表示“这个 user 在这个 cutoff 时刻的预测样本”。同一个 user 在不同 cutoff 上理论上可以出现多次。

DFS 必须保证最终特征仍然回到这个 target grain。否则如果裸 JOIN orders，让一个 `(user, cutoff)` 复制成几十行，会产生三个问题：

1. 标签被重复；
2. 订单多的用户在 loss 中被隐式加权；
3. 模型实际学到的 prediction unit 已经偷偷从 user-level 变成了 user-order-level。

这也是理解 relational flattening 最重要的一点：**flatten 不是“把 schema 消掉”，而是把关系传播出来的历史信息压回目标预测粒度。**

### DFS 与 feature extras

官方 `TabPFNRelModel.fit()` 的主路径非常直接：

```text
build_dfs_features(...)
FeaturePipeline.fit_transform(...)
ContextStrategy.fit(...)
```

`FeaturePipeline` 还能在 DFS 之后附加三类额外信号：

- calendar features：cutoff timestamp 的周期性 sin/cos；
- history lags：同一 entity 的历史 target lag 与 age；
- raw text：anchor/entity table 上的字符串列重新 attach 回 flat table。

其中 history lag 使用严格早于 cutoff 的历史记录，代码里明确使用 `allow_exact_matches=False` 来避免当前 anchor 看到自身标签。

### Context selection：不是简单 random 100k

TabPFN 是 in-context learning，所以“给它哪些训练样本”本身就是算法的一部分。

代码支持三种 context strategy：

- `random`：uniform seeded downsample；
- `hard_pool`：先取最近的 `M` 行，再让每个 estimator 从中无放回抽 `K` 行；
- `soft_pool`：按 recency-decay 权重采一个 `M` 行 pool，再从 pool 中给不同 estimator 抽 `K` 行。

论文 release 配置使用 `hard_pool`，`K = 100k`，`M = 4K`，默认 8 个 estimators。这个设计的重点是同时保留：

- temporal recency：近期样本通常更接近测试分布；
- ensemble diversity：不同 estimator 不完全看到同一批 context。

因此“TabPFN-3 能塞更多 rows”并不是全部，**context 怎么选**也是 TabPFN-Rel 相比旧 RDBLearn recipe 的实质方法改动之一。

### Local vs API：有没有一套 TabPFN-Rel 专属权重？

从代码实现看，**没有一套独立的 `TabPFN-Rel` checkpoint**。

两条路径共享 relational harness，只在 TFM backend 上分叉：

- `tabpfn-rel-local`
  - `tfm = tabpfn-v3`
  - 通过本地 `tabpfn` package 调 `TabPFNClassifier/Regressor.create_default_for_version(ModelVersion.V3)`；
  - 不支持 raw text；
  - relational DFS / context selection 都在本地代码里。
- `tabpfn-rel-client`
  - `tfm = tabpfn-v3-api`
  - 通过 `tabpfn_client` 调 hosted TabPFN-3，`model_path="v3_default"`；
  - fit / predict server-side；
  - 支持 raw text columns。

所以比较准确的说法是：

> Relational 部分已经以代码形式公开；TabPFN-Rel 本身不是一套藏在 API 里的专属模型权重。完整 text-capable 路径依赖 hosted API，而 text-free 的 TabPFN-3 路径可以本地跑。

另外，RelArena 代码本身明确把本地 TabPFN dependency 标记为 Prior Labs License，而不是普通 permissive dependency；“能本地跑”和“权重完全按 Apache/MIT 方式开放”不是一回事。

## 关键实验结果

官方 release snapshot 在 21 个 RelArena-α tasks 上给出的单 seed Elo：

| Method | Kind | Model board | Model + system board |
| --- | --- | ---: | ---: |
| RT-PluRel | system | — | 1861 |
| TabPFN-Rel (API) | model | 1821 | 1826 |
| TabPFN-Rel (OSS) | model | 1706 | 1727 |
| GraphSAGE | model | 1658 | 1655 |
| RelGT | model | 1575 | 1584 |
| RDBLearn | model | 1548 | 1554 |
| RelGNN | model | 1506 | 1519 |
| Constant (per-entity) | model | 1256 | 1256 |
| Constant (global) | model | 1000 | 1000 |

这组结果最值得看的不是“TabPFN-Rel 第一”本身，而是三个更有普适意义的现象：

1. aggregation-based tabular 方法依然很强；
2. 一个几乎不建模结构的 per-entity constant baseline 都能在部分任务上赢 RelGNN / RelGT，说明 benchmark task 里 persistence / entity prior 很强；
3. relational benchmark 的整体 compute 成本非常高，预处理往往比模型 forward 更麻烦。

注意：这是 release snapshot、单 seed、Elo 聚合结果，不应该把几十个 Elo 点的差距解读成已经非常稳定的算法定论。

## 关键分析

### 贡献重心确实偏工程与 benchmark

如果粗略拆贡献重心，我们当前的主观判断大约是：

- 60% benchmark / reproducibility / engineering infrastructure；
- 25% empirical finding；
- 15% TabPFN ecosystem extension / product positioning。

这个比例只是我们的阅读判断，不是论文自己的 claim。

为什么会有这种感觉：RelArena 仓库真正复杂的部分大量集中在 protocol、runner、search space、data state、cache、RPI、baseline integration，而不是在 `tabpfn_rel/model.py` 里实现一个很复杂的新 neural architecture。

### 但不能简单归类成“只有包装”

TabPFN-Rel 至少有几处会实质改变结果的方法层改动：

- 更严格一致的 temporal tuning database state；
- TabPFN-3 backbone 与更大的 context；
- recency-aware context selection；
- validation examples 进入 test-time context 的策略；
- anchor text re-attachment 与 hosted text handling；
- leak-aware history features。

所以它更准确的位置是：**algorithmic novelty 不在新的 relational architecture，而在 pipeline / protocol / context / feature handling。**

## Insights

### Insight 1：关系数据未必首先需要关系神经网络

这篇真正挑战的是一个社区默认假设：

> 既然数据是 relational 的，是不是必须用 GNN / relational Transformer 才算“利用关系”？

RelArena 的结果至少说明，在当前 entity-level forecasting 任务上，**强 relational feature engineering + 强 tabular learner** 是不能被跳过的 baseline。

如果一个复杂 relational architecture 打不过它，问题可能不是“TabPFN 太强所以不公平”，而是需要回答：你的结构 inductive bias 到底带来了什么额外可泛化信息？

### Insight 2：target grain 比 JOIN 技巧更重要

relational ML 最容易被低估的问题之一是 prediction grain。

先明确 `(entity, cutoff)` 是什么，再讨论 join、graph、aggregation、attention。否则很多所谓 relational feature 很可能只是改变了样本权重或偷偷引入 future information。

### Insight 3：benchmark protocol 本身就是方法的一部分

在 temporal relational task 里：

- 数据库冻结在哪个时间；
- validation 时能看到哪些历史；
- test 时能否把 validation labels 当 context；
- preprocessing cache 是基于哪个 history 构建；

这些都可能比换一层 GNN block 更影响最终分数。

### Insight 4：强 constant baseline 暗示任务的 persistence 很重要

per-entity constant 能打赢部分 relational deep models，说明一些任务里 entity-specific prior、历史惯性或者 label imbalance 很强。

因此后续看新的 relational paper 时，应该特别检查：

- 是否报告 global constant；
- 是否报告 per-entity / last-value baseline；
- 是否真的超过 entity-only tabular baseline；
- 复杂模型的收益是否来自结构，而不是更好的 temporal leakage handling。

## 我们的观点

### “主要是工程工作，再把自家模型捧一下”——基本成立，但要加限定

这个判断有事实基础：

- headline 同时围绕 RelArena、TabPFN-Rel、RPI 三个自家组件展开；
- TabPFN-Rel 的 backbone 是 TabPFN-3；
- release 结果突出其 model-board 第一；
- 核心代码显示 relational method 本身主要是 DFS + feature/context pipeline + TabPFN backend。

但如果因此把论文完全归为 marketing，也会漏掉两件重要的事：

1. 统一 temporal benchmark / tuning protocol 本身确实有研究价值；
2. “flattening + strong tabular model 依旧 competitive”是一个值得社区认真验证的 empirical claim。

### 对我们来说值不值得深挖？

如果目标是找新的 tabular architecture / pretraining objective，这篇优先级不高，因为它没有提出一个新的 TabPFN 类训练范式。

如果目标是理解：

- tabular foundation model 怎么进入真实多表数据库；
- relational benchmark 怎么做 leak-safe evaluation；
- GNN 与 feature-engineering baseline 到底应该怎么公平比较；
- entity-level temporal task 应该怎么定义；

那么这篇非常值得保留。

## GitHub / Code Analysis

### 仓库结构

官方仓库的大体结构：

```text
relarena/
├── src/relarena/
│   ├── model.py
│   ├── search_space.py
│   ├── registry.py
│   ├── tasks.py
│   ├── metrics.py
│   ├── tuner.py
│   ├── runner.py
│   ├── results.py
│   ├── cache.py
│   ├── models/
│   ├── featurization/
│   ├── evaluation/
│   └── userdb/
├── baseline_results/
├── docs/
├── examples/
├── workflows/
└── tests/
```

TabPFN-Rel 自己其实很小：

```text
src/relarena/models/tabpfn_rel/
├── __init__.py
├── context.py
├── features.py
└── model.py
```

这反过来说明了论文的性质：**复杂度主要在 benchmark/runtime/data plumbing，不在 TabPFN-Rel neural model class。**

### TabPFN-Rel 的实现路径

`model.py`：

- 定义 local/client 两个注册模型；
- search space 只主要扫 DFS depth；
- `fit()` 负责串起 DFS、feature extras、context strategy 和 TFM；
- `predict()` 做同样的 DFS + transform 后直接调用 fitted TFM。

`features.py`：

- cutoff calendar feature；
- target history lag；
- anchor table raw text lookup 与 re-attach；
- train / val / test schema 对齐。

`context.py`：

- random / hard_pool / soft_pool；
- hard pool 直接按 cutoff 排最近 `M` 个；
- soft pool 用随 recency 衰减的权重采样；
- 不同 estimators 从 pool 中取不同 context，并对它们的 union 做 fit。

共享 `featurization/dfs.py`：

- 真正负责 relational DFS；
- 最大深度 feature matrix 可被较浅 depth config 切片复用；
- process-level 与 on-disk cache 都在这一层处理；
- TabPFN-Rel 与 RDBLearn 输入一致时可以共享 leak-safe DFS matrices。

### Cache 是这个仓库里非常实际的一部分

官方明确指出 DFS 是 CPU / RAM intensive，并允许提前 warm preprocessing cache。

缓存策略不是把一个最终 feature CSV 随便存下来，而是围绕：

- split database state；
- history input；
- DFS depth；
- artifact version/key；

做可重建的 Parquet cache。

官方 demo `examples/tabpfn_rel_caching.py` 在 `rel-f1/driver-dnf` 上给的示例约从 409s 降到 12s，并检查 cached / uncached features 或 predictions 一致。

这也是一个很好的工程判断：

> 论文 headline 看起来在讲“模型”，但真实 relational ML pipeline 的主要成本可能发生在模型 fit 之前。

### 工程质量判断

优点：

- model registry / search space / runner 的边界比较清楚；
- temporal split 与 cache key 有意识地围绕 leakage 设计；
- baseline 和系统 submission 共用统一评估框架；
- cache warmer 是公开可重建的，而不是只有作者拥有的私有 artifact；
- TabPFN-Rel 实现足够短，方法路径容易审计。

需要注意：

- α-release，官方明确不是 production-ready；
- RPI 当前可表达任务范围有限；
- preprocessing runtime 没有完全进入 timed model experiment，因此端到端成本比较要谨慎；
- API 版依赖外部 hosted service，text 路径无法做到完全纯本地复现；
- 单 seed release Elo 适合看大趋势，不适合过度解释细小排序差异。

## 值得继续追的问题

1. 在更多 relational benchmark / 非 entity forecasting 任务上，DFS + tabular model 还能不能维持优势？
2. TabPFN-Rel 相比 RDBLearn 的增益里，TabPFN-3 backbone、context selection、tuning protocol 各自占多少？
3. 如果加入更强、经过认真 tuning 的 non-TabPFN tabular baseline，排名会怎样？
4. DFS 在高 fan-out、长 join path 上的 compute / feature explosion 是否会成为真实部署瓶颈？
5. relational GNN 真正应该赢的 task family 是什么——需要哪些无法被 aggregation summary 保留的结构信号？
6. RPI 能否逐步扩展到 edge-level、link prediction、multi-entity target 或更一般的 temporal relational task？
