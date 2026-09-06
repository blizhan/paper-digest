---
type: Research Paper Review
title: "DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces"
description: "Data Agent benchmark 与系统设计分析，重点提炼 target contract、workspace catalog、join invariants、lineage、verification 和 harness 设计经验。"
resource: https://arxiv.org/abs/2608.03451
tags: [data-agent, benchmark, heterogeneous-workspace, verifiable-analytics]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2608.03451
    title: "DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces"
  - id: code
    resource: https://github.com/HKUSTDial/DataSpace
    title: HKUSTDial/DataSpace
---

# DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces

## Links

- Paper: [arXiv:2608.03451](https://arxiv.org/abs/2608.03451)
- PDF: [arXiv PDF](https://arxiv.org/pdf/2608.03451)
- Hugging Face Papers: [2608.03451](https://huggingface.co/papers/2608.03451)
- Project / leaderboard: [dataspace-bench.github.io](https://dataspace-bench.github.io/)
- Official code: [HKUSTDial/DataSpace](https://github.com/HKUSTDial/DataSpace)
- Dataset: [HKUSTDial/DataSpace](https://huggingface.co/datasets/HKUSTDial/DataSpace)
- KDD Cup 2026 Data Agent Track: [dataagent.top](https://dataagent.top/)
- 本笔记代码分析基于官方仓库 commit `6491caa4c70cc06cacb6103ba73cefb00746abfe`。

## 一句话结论

这篇最值得看的不是某个新 agent architecture，而是它试图把“一个 Data Agent 到底算不算把活干完了”定义清楚：给 agent 一个包含 CSV、JSON、SQLite、Markdown、PDF、video 的 task-local workspace，让它自己发现证据、跨 artifact 对齐和 join、完成计算，最后必须交付一张**完整、可程序化验证的结果表**。

对我们更有价值的是实验暴露出的系统性问题：固定 MiMo-V2.5 只换 harness，accuracy 可以从 30.98% 到 46.34%，差 **15.36 个百分点**；对最强 backbone Grok 4.5 的 136 个失败做 trace audit 时，**71/136（52.2%）根因在 answer materialization**。因此 Data Agent 目前很大程度上还是一个 systems problem：target schema、workspace navigation、join semantics、execution state、context management、result verification 都是核心算法的一部分。

## 文章摘要

DataSpace 面向 heterogeneous workspace 上的可验证数据分析。每个 task 给一个自然语言问题和一个自包含 workspace，agent 需要自行定位相关 artifact，跨语言、跨 schema、跨 modality 对齐证据，执行 filter / join / aggregate / rank / temporal reasoning，并输出完整的 tabular result。

Benchmark 包含：

- 410 个任务；
- 7,439 个 artifacts；
- 15.01 GB workspace 数据；
- CSV、JSON、SQLite、Markdown、PDF、video 六类 modality；
- finance、macroeconomics、healthcare 等领域；
- 265/410（64.6%）cross-language tasks；
- 134/410（32.7%）任务的 verified solution path 需要组合多个 modality；
- 113/410（27.6%）任务需要 join。

作者用 `DataSpace-Builder` 从可执行 Text-to-SQL 资源出发，经 cross-language transformation、constraint-aware relational sampling、modality routing / rendering 和 human review / repair 构造任务。410 个 task 中 363 个（88.5%）来自 BULL，47 个（11.5%）来自 EHRSQL。

评测不让 LLM judge 一段自由文本，而要求 agent 产出 `prediction.csv`。官方 evaluator 对列做 header-invariant alignment，并按类型、precision、unit 和 row-order semantics 做 deterministic comparison。410 个 input 全部公开，其中 60 个 task 同时公开 gold result 和 frozen evaluation config，其余 350 个 reference withheld 用于完整官方评测。

## DataSpace 到底在测什么

作者把任务拆成四类能力：

```text
workspace discovery
    ↓
interpretation / alignment
  - type
  - schema
  - entity
  - unit
  - language
    ↓
relational computation
  - filter
  - join
  - aggregation
  - ranking
  - temporal reasoning
    ↓
complete tabular-result materialization
```

这里最后一步很关键。它不接受“我已经知道答案大概是什么”，而是要求 agent 把用户要的 relation 完整做出来。少一行、多一行、少一列、grain 错、单位错、排序错、重复没处理好，都可以被 evaluator 暴露出来。

因此 DataSpace 测的不是单纯 `reasoning correctness`，而更接近：

```text
semantic correctness
× data access correctness
× relational execution correctness
× output-contract correctness
```

只要其中一个环节失败，task 就失败。

## Benchmark 构造

### DataSpace-Builder

论文的四阶段构造流程是：

```text
executable Text-to-SQL instance
        ↓
Cross-Language Transformation
        ↓
Constraint-Aware Relational Sampling
        ↓
Modality Routing & Artifact Rendering
        ↓
Human Review & Task Repair
        ↓
heterogeneous task-local workspace
```

这里有两个设计点值得注意。

第一，sampling 不是独立对每张表 random sample。作者会保留 PK/FK、已知 join path、predicate value、boundary value、target entity 等 safeguards，再沿关系传播 key，避免 sampling 把查询需要的连接关系破坏掉；sample 后重新执行 SQL，得到新的 candidate reference。

第二，modality rendering 不是简单把 CSV 转格式。Markdown / PDF 会通过生成流程把表记录变成文档，video 还可能是 query-conditioned：从 SQL AST 和 sampled result 中选 filter condition / result cell 等 evidence，再放进视频场景，并相应修改问题。也就是说 benchmark 的多模态证据是经过受控构造的，而不是原封不动采集自真实企业 workspace。

所有 candidate 最后由 11 位 domain experts 中的两位独立 reviewer blind solve，再看 gold 并做 repair / reject。

## Deterministic evaluator 为什么重要

官方 evaluator 的核心不是比 header 名字，而是比 relation semantics。

当前实现会先要求预测和 gold 的列数、行数一致，然后对预测列尝试 one-to-one permutation；每个候选映射按 gold column 的类型配置做 normalization，最后比较行关系：

- order-sensitive task：要求 canonical row sequence 完全一致；
- order-insensitive task：用 row multiset 比较，因此 duplicate count 也必须一致；
- number：支持 integer / decimal places / significant digits 等比较模式；
- 还能配置 percentage / fraction 等 unit semantics；
- date / datetime / boolean 也有类型归一化。

这比 LLM-as-a-judge 更适合 analytics benchmark，因为它能稳定区分：

```text
“大概算对了”
vs
“交付了正确的完整 relation”
```

同时它也比只评 SQL execution result 更接近 agent 工作流，因为 agent 不会被直接告知哪张数据库、哪段 SQL 或哪种 modality 才是答案来源。

## 关键实验结果

### 1. Backbone 仍然重要，但 benchmark 明显没有饱和

固定 `DataSpace-Agent` harness，在 410 个任务上的结果：

| Backbone | Correct | Task Accuracy |
| --- | ---: | ---: |
| Grok 4.5 | 272 | 66.34% |
| GPT-5.6 Sol | 265 | 64.63% |
| Kimi K3 | 219 | 53.41% |
| MiMo-V2.5 | 161 | 39.27% |
| Claude Sonnet 5 | 135 | 32.93% |
| MiniMax M3 | 117 | 28.54% |

六个模型里有 76 个 task 没有任何一个做对；oracle union 也只有 334/410（81.46%）。因此它不是一个已经接近 ceiling 的 benchmark。

### 2. Harness 的影响大到不能忽略

固定 MiMo-V2.5，只换 harness：

| Harness | Correct | Task Accuracy |
| --- | ---: | ---: |
| Grok Build | 190 | 46.34% |
| Claude Code | 183 | 44.63% |
| DataSpace-Agent | 161 | 39.27% |
| Codex | 143 | 34.88% |
| Smolagents | 127 | 30.98% |

最高和最低相差 **15.36 pt**。

因此一个 Data Agent 的 end-to-end 能力更接近：

```text
foundation model
× planner / harness
× tool-use policy
× context management
× execution runtime
× output verification
```

不能把 Data Agent leaderboard 直接解读成纯 foundation-model reasoning 排名。

### 3. 更长的 trajectory 不等于更强

固定 DataSpace-Agent，GPT-5.6 Sol 只有比 Grok 4.5 低 1.71 pt，但平均使用：

- 74.2% 更少的 token；
- 50.3% 更少的 tool actions；
- 39.2% 更少的 wall-clock time。

这说明 Data Agent 不应该只报 accuracy。至少还应同时报告 token、actions、latency、cost，否则无法区分“更聪明”与“只是多试很多次”。

### 4. Join 和 cross-modality integration 是最稳定的难点

在六个 backbone 上：

- multimodal tasks 比 single-modal tasks 低 1.8–14.0 pt；
- join-required tasks 低 9.7–19.8 pt。

论文也强调这些 stratified gaps 是 descriptive，而不是 causal proof。

这里的 join 难点不能简化成“模型不会写 SQL JOIN”。真实 failure surface 更接近：

```text
schema alignment
  ↓
entity / key alignment
  ↓
join cardinality
  ↓
semantic filter / temporal scope
  ↓
aggregation grain
  ↓
final projection
```

SQL syntax 往往只是其中最容易的一层。

### 5. 最大失败源竟然是 materialization

作者对 Grok 4.5 的 136 个失败任务做 trace-level audit：

- answer materialization：71/136（52.2%）；
- task specification / intent：31/136（22.8%）；
- 其中 60 个 materialization failures 是在内部已经拿到所需结果后，最终又多加或漏掉列；
- 17 个 intent failures 是一开始就误解了 requested output 或 row grain；
- 这两类与 answer schema 直接相关的问题合计 77/136（56.6%）；
- 真正因为“选错 evidence source”的只有 3 个。

这个结果非常重要：

> 很多 Data Agent 不是“不会算”，而是“没有稳定地把正确结果交付出来”。

## 对设计 Data Agent 最值得复用的原则

下面不是作者逐条提出的 architecture，而是我们根据 DataSpace 的 benchmark finding、failure audit 和官方 baseline 代码推导出的设计原则。它们应当和论文事实区分开。

### 原则 1：先定义 target grain 和 output contract，再开始找数据

DataSpace 的 failure audit 直接显示，row grain / requested output 理解错误和最终 column projection 是最主要的失败来源。

因此 planner 的第一阶段不应该是立刻 `ls` / `SELECT` / 打开 PDF，而应该先把问题编译成一个显式 target contract，例如：

```text
target entity: customer
target grain: one row per customer
required columns:
  - customer_id
  - avg_monthly_spending_2025
ordering: customer_id ASC
units: CNY
null policy: keep customers with no transactions as 0
```

这个 contract 后面既指导查询，也直接成为最终 verifier 的检查表。

一个很实用的 mental model 是：

> Data Agent 的 planner 先生成“目标 relation specification”，再生成执行计划。

### 原则 2：把 LLM 当 semantic planner，不要当 dataframe engine

LLM 更适合决定：

- 哪两个字段表达同一实体；
- 某段文档定义了什么业务口径；
- 哪个 filter 对应用户意图；
- 最终 grain 是 account、customer 还是 customer-month；
- 某个时间表达应该映射到哪个 fiscal period。

而下面这些应该尽量交给 deterministic engine：

- join；
- filter；
- groupby；
- sort；
- deduplicate；
- numeric / date conversion；
- unit conversion；
- exact table materialization。

即：

```text
LLM = semantic query planner
SQL / DuckDB / pandas = execution engine
```

不要让模型在 context 里“心算 dataframe”。

### 原则 3：先建立 workspace catalog，再做 selective deep inspection

DataSpace 每个 task 中位数约 20 个 artifact，而且“存在于 workspace”不代表“是 verified path 需要的 evidence”。例如 CSV 在每个 workspace 都存在，但 verified solution path 只在 58 个 task 中需要 CSV。

因此更合理的 harness 应先构建轻量 catalog，例如：

```text
file
type
size
schema / columns
candidate primary keys
candidate foreign keys
time columns
short semantic summary
possible relations
```

之后 planner 基于 catalog 选择需要深读的文件、PDF pages、video timestamps 或表，而不是在 workspace 里 random walk。

这里的目标不仅是 accuracy，也是减少 token / action / context pollution。DataSpace 的 efficiency result 已经说明，near-top accuracy 不要求更长的 trajectory。

### 原则 4：Join 必须有显式 cardinality 和 grain checks

DataSpace 中 join 是跨 backbone 最稳定的 difficulty factor。我们的工程推导是：任何重要 join 都不应该只执行，不做验证。

join 前至少记录：

```text
left rows
right rows
left key uniqueness
right key uniqueness
null ratio on join keys
expected relationship: 1:1 / 1:N / N:1 / N:N
```

join 后检查：

```text
row count delta
unmatched key count
duplicate expansion
target grain uniqueness
```

例如一个原本应该 `one row = one customer` 的中间表，join 后 customer 出现几十次，agent 必须显式判断这是预期的一对多展开，还是 accidental multiplicative join。

### 原则 5：每个重要 intermediate table 都要带 invariants

不要把 tool call “退出码为 0”当作数据正确。

对中间结果可以持续维护：

```text
row count
unique target-key count
duplicate ratio
NULL ratio
date range
numeric range
units
aggregation totals
schema
```

这些 invariants 本质上是把传统 data engineering 的 data-quality check 放进 agent loop。相比单纯增加更多 CoT，这种机制更直接针对 DataSpace 暴露的 execution failure。

### 原则 6：保留 lineage，让结果能回溯到 evidence

跨文件、跨 modality 任务最危险的不是单个数算错，而是 agent 后来已经不知道一个数来自哪里、在哪一步被变换过。

推荐把 intermediate result 附带最小 lineage：

```text
column / derived field
  ← source artifact
  ← source field / page / timestamp
  ← transformation
  ← join key
```

这能同时帮助：

- debug schema / entity alignment；
- 在 final verification 时抽样回溯；
- context compaction 后保留关键 provenance；
- 向用户解释结果而不用重新搜索整个 workspace。

DataSpace 本身没有要求 agent 输出 lineage，这是我们基于 cross-artifact failure surface 的设计建议。

### 原则 7：materialization 要有独立 verification pass

这是 DataSpace 最直接的工程教训。

不应该：

```text
compute dataframe
  ↓
save csv
  ↓
done
```

而应该：

```text
compute result
  ↓
materialize prediction.csv
  ↓
重新读取文件
  ↓
verify against target contract
  - column count / semantics
  - grain
  - uniqueness
  - duplicates
  - required rows
  - units / precision
  - ordering
  - null policy
  ↓
sample lineage backtrace
  ↓
submit
```

最好把 planner / executor 和 final verifier 角色分开，类似 coding agent 的：

```text
write code → run tests
```

Data Agent 应该是：

```text
compute table → validate table
```

### 原则 8：context window 不是 RAM，要有外部 working state

15.36pt harness gap 说明 agent harness 的 context / state management 本身会显著影响结果。

不应该反复把这些东西全部塞进 prompt：

```text
完整 CSV
完整 SQL output
PDF 全文
所有历史 tool calls
巨大 dataframe preview
```

更合理的是：

```text
raw artifact
  ↓
metadata / catalog
  ↓
selected slice
  ↓
named intermediate table
  ↓
compact state summary + lineage
```

Agent 应该把文件、SQLite / DuckDB、intermediate parquet / CSV、structured state 当外部 working memory，用 context window 保存“决策状态”，而不是保存全部原始数据。

### 原则 9：harness 本身是一等研究对象

同一个 backbone 差 15.36 pt，已经足够说明不能把 harness 当薄薄的 wrapper。

以后评 Data Agent paper，应至少问：

- model 是否固定？
- system prompt 是否固定？
- tool set 是否固定？
- runtime / package 是否固定？
- network 是否一致？
- action / wall-clock budget 是否一致？
- context compaction 是否一致？
- 是否允许 subagent / memory / retry？
- final output validation 是否一致？

不控制这些变量，“模型 A 比模型 B 更会做数据分析”的结论可能非常不稳。

### 原则 10：评测必须同时看 correctness 和 efficiency

只报 accuracy 会鼓励无限加 trajectory、retry 和 token。

至少应该同时报告：

```text
task accuracy
tokens
tool actions
wall-clock latency
API / compute cost
failure type
```

DataSpace 中 GPT-5.6 Sol 与 Grok 4.5 的 trade-off 就说明：接近最高准确率的 agent 可以有完全不同的资源曲线。

## 我们当前更倾向的 Data Agent architecture

综合这篇的 finding，我们更倾向把 Data Agent 设计成“带 LLM planner 的小型 data operating system”，而不是“装了 Python tool 的 chatbot”。

一个理想化结构是：

```text
                  ┌─ workspace inventory / catalog
                  │
User question ─→ target-contract compiler
                  │
                  ↓
               planner
                  │
         ┌────────┼────────┐
         ↓        ↓        ↓
 schema/entity  document   execution engine
 alignment      evidence   SQL / DuckDB / pandas
         └────────┼────────┘
                  ↓
         named intermediate tables
          + invariants + lineage
                  ↓
             result builder
                  ↓
          independent verifier
                  ↓
              final table
```

其中应该把这些东西当 first-class object：

```text
target grain
target schema
schema / entity relations
lineage
intermediate state
invariants
verification
materialization contract
```

比“再加一个更长 prompt / 更多 CoT”更值得优先投资。

## 关键分析

### 贡献重点是 benchmark contract，不是新 agent algorithm

这篇最扎实的贡献是把四件事绑在一起：

1. heterogeneous workspace discovery；
2. structured / unstructured / video evidence integration；
3. complete tabular output contract；
4. deterministic, model-free evaluation。

它不是靠一个复杂的新 architecture 赢 benchmark。官方 `DataSpace-Agent` 反而刻意保持简单，方便固定 agent 后比较 backbone。

### 这个 benchmark 同时也是 systems benchmark

15.36pt harness spread 是最应该记住的数字之一。

它说明 Data Agent 的效果不是单个模型 checkpoint 的属性，而是 model 与 planner、tools、context、runtime、verification 的组合属性。以后如果论文只报告“某个 agent + 某个模型”总分，而不拆 harness 条件，解释空间会非常大。

### Output schema 是一个被严重低估的 reasoning object

56.6% 的 audited failures 最终和 target-result misunderstanding / faulty column projection 有关。这个 finding 很值得单独抽象出来：

> 最终 output schema 不应该只是最后 `to_csv()` 时才决定的格式，而应该从 task planning 开始就作为核心状态被维护。

这也是为什么我们把 `target grain + target schema` 放在设计原则第一位。

## Benchmark 局限

### 1. “真实 workspace”仍然是 synthetic-but-reviewed

任务起点是 BULL / EHRSQL 的 executable Text-to-SQL instance，再通过 sampling、translation 和 modality rendering 构造 workspace。

这比纯 Text-to-SQL 更接近真实 data work，但不能直接等价于企业里的 messy workspace。真实环境常见的：

- 重复和过期文件；
- conflicting versions；
- schema drift；
- undocumented fields；
- permission boundary；
- partial failure；
- 缺失、错误、手工修过的数据；
- 部门间互相矛盾的指标定义；

并不是当前 benchmark construction 的主要对象。

尤其 video 可能是从 SQL/query evidence 受控生成的，因此“真实视频理解难度”和这里的 video evidence 难度也不能直接画等号。

### 2. 410 tasks 对异构空间仍然不大

对一个同时横跨 modality、language、join pattern、answer shape、domain 的 benchmark，410 个任务很难覆盖真实 workload 的长尾。

因此按 task subgroup 分层看到的 gap 很有价值，但不应该过度解释成稳定的因果规律。

### 3. Table-only output 是优点，也是边界

要求完整 relation 能做 deterministic evaluation，是这篇最强的设计之一。

但它也意味着 DataSpace 评估的是 **verifiable analytical tasks** 的一个重要子集，而不是所有 data-agent / knowledge-worker 工作。例如探索性分析、可视化选择、因果讨论、策略建议、交互式澄清就不适合直接压成一张 gold table。

### 4. Harness gap 说明问题存在，但没有完全定位问题

固定 backbone 后的 15.36pt 差距证明 harness 很重要，但它把 planning、memory、tool strategy、context compaction、retry policy 等很多因素一起改变了。

因此它是很强的 systems-level finding，却不是对“哪一种 harness mechanism 最有效”的 ablation。

### 5. Public input + withheld gold 仍需长期关注 contamination / overfitting

所有 410 个输入公开，60 个 reference 公开，350 个 gold withheld。这对当前 leaderboard 是合理折中，但随着 benchmark 被长期使用，仍需要关注：

- agent/harness 对公开 input 的 pattern overfitting；
- model pretraining contamination；
- hidden set 是否需要定期 rotation；
- benchmark-specific workflow 是否逐渐替代通用 data-agent ability。

## Insights

### Insight 1：Data Agent 的关键瓶颈已经从“能不能算”扩展到“能不能可靠交付”

`materialization` 是一等能力，而不是最后一行 `to_csv()`。

### Insight 2：target grain / output schema 应该进入 planner state

它和 SQL plan、tool plan 一样重要，而且应该贯穿 execution 和 verification。

### Insight 3：Agent benchmark 很大程度上也是 systems benchmark

固定 backbone 的 15.36pt harness spread 足以说明 planner、runtime 和 context policy 可以和换模型一样重要。

### Insight 4：Join failure 的核心是 relational semantics，不是 SQL syntax

真正难的是 schema / entity alignment、cardinality、grain、temporal scope 和 aggregation composition。

### Insight 5：更长 trajectory 不是默认答案

准确率接近时，token/action/latency 可以差非常大。更好的 state compression、tool planning 和 deterministic execution 往往比无脑扩大 agent loop 更有价值。

### Insight 6：数据库与 data engineering 的经典机制值得重新进入 agent architecture

query planning、schema catalog、provenance、constraint checking、data-quality invariants、transaction-like execution state、verification，这些可能比纯 prompt trick 更接近 Data Agent 的长期核心能力。

## 我们的观点

我们目前认为 DataSpace 是一篇**benchmark / systems 价值明显高于新模型算法价值**的工作，但这不是贬义。相反，Data Agent 这个方向现在最缺的恰好是靠谱的 task contract 和 failure signal。

这篇最值得长期保留的不是 66.34% 这个 leaderboard 数字，而是：

```text
15.36 pt harness gap
52.2% materialization root causes
56.6% output-schema-related audited failures
9.7–19.8 pt join gap
```

这些数字共同支持一个判断：

> 好的 Data Agent 更像一个带 LLM planner 的 query engine / data operating system，而不是一个会调用 Python 的聊天机器人。

如果我们自己做 Data Agent，优先级会放在：

1. target contract / grain；
2. workspace catalog；
3. schema + entity alignment；
4. deterministic execution；
5. intermediate invariants + lineage；
6. context / state management；
7. independent result verifier；
8. correctness × efficiency 的统一评测。

## GitHub / Code Analysis

本节基于官方仓库 commit `6491caa4c70cc06cacb6103ba73cefb00746abfe` 的实际代码阅读。

### 仓库结构

当前官方仓库主要公开两块：

```text
evaluation/
  evaluate.py
  schema/
  tests/

baseline/
  src/dataspace_agent/
  src/dataspace_baselines/
  configs/
  docker/
  tests/
```

Benchmark 数据本身通过 Hugging Face dataset 单独发布。

需要注意：论文中的 `DataSpace-Builder` 是重要贡献，但**当前这个 GitHub commit 没有包含完整 benchmark construction / rendering pipeline 的源码目录**。因此我们可以代码审 evaluator 和 baseline agent，但不能从公开仓库代码完整复核 DataSpace-Builder 的生成实现。

### DataSpace-Agent：刻意保持最小工具面

核心入口是：

```text
baseline/src/dataspace_agent/agent.py
baseline/src/dataspace_agent/tools.py
```

`agent.py` 实现一个 ReAct-style model/tool loop。系统 prompt 明确要求：

- 只能使用 `/workspace`；
- workspace read-only；
- intermediate 和 final output 写 `/output`；
- 最终必须生成 `/output/prediction.csv`；
- 最终必须调用 `submit_answer`；
- prose answer 不算提交。

工具只有三个：

```text
bash(command, timeout)
view_image(path)
submit_answer(path)
```

`tools.py` 也明确说明 `view_image` 只返回 agent 自己选出的单张图；PDF page / video frame 需要先通过 bash 自己 render / extract。工具层没有 OCR retrieval、document QA、NL2SQL、schema linker 或自动 evidence selection。

这点非常重要：官方 baseline 并没有藏一个强数据专用 solver。

### Runtime：generic data workbench，而不是 benchmark-specific toolkit

`baseline/docker/workbench/capabilities.json` 中当前 runtime 提供的主要是通用本地工具：

- pandas / numpy / pyarrow / jq / rg；
- SQLite；
- pdftotext / fitz / pypdf / pdfplumber；
- image / OCR utilities；
- ffmpeg / ffprobe / cv2。

同时：

- runtime offline；
- package installation disabled；
- `asr: false`。

baseline README 明确写了没有 semantic retrieval、schema linking、document QA、video QA、NL2SQL、modality routing 或 automatic evidence selection。

因此 harness comparison 更接近比较各 harness 原生的 planning / tool use / memory / context behavior。

### Harness comparison 的公平性边界

`baseline/configs/harness.yaml` 当前把：

- backbone；
- wall-clock deadline；
- shared workbench runtime；
- task input；
- output contract；

统一起来，同时允许各 harness 保留 native planning、tool APIs、context compaction 等行为。

默认 harness experiment 配的是 MiMo-V2.5、1800 秒 deadline，并固定 Codex / Claude Code / Grok Build / Smolagents 的版本。

这意味着 15.36pt harness gap 是一个合理的 end-to-end systems comparison，但并不是严格的单变量 ablation：不同 harness 的 planner、memory、tool interface、compaction 仍然一起变化。

### `submit_answer` 只做格式验证，不泄露 correctness

`baseline/src/dataspace_baselines/core/prediction.py` 中的 `validate_prediction()` 只检查：

- 文件存在；
- 有 header；
- 至少一列；
- 每行列数一致；
- 计算 rows / bytes / SHA256 等 metadata。

它不会告诉 agent 结果是否正确。

这和 benchmark 设计是对齐的：agent 不能靠 evaluator feedback 反复 trial-and-error 拟合 gold。

### Official evaluator 的实际比较路径

`evaluation/evaluate.py` 的逻辑比 README 描述更值得看。

主要路径是：

```text
strict CSV parse
  ↓
exact column-count check
  ↓
exact row-count check
  ↓
normalize prediction columns under each gold type spec
  ↓
try one-to-one prediction-column permutations
  ↓
canonical row comparison
  ├─ exact ordered sequence
  └─ unordered Counter multiset
```

因此所谓 header-invariant 不是模糊 semantic matching header，而是：只要预测列能找到一个类型和数据都匹配的 one-to-one mapping，就可以通过。列名和列顺序本身不计分。

这个实现非常 deterministic，也很容易复核。

### License / reproducibility

官方代码仓库当前是 MIT License。

baseline 还包含：

- runtime / harness version pinning；
- execution sandbox；
- network isolation；
- run logs；
- resume identity / fingerprint；
- prediction acceptance tests；
- evaluator tests。

整体上这部分工程做得比较认真，尤其适合作为后续自己搭 Data Agent benchmark 时的 protocol 参考。

## 值得继续追的问题

1. **把 harness gap 进一步拆开。** planning、context compaction、tool interface、retry、memory 各自贡献多少？目前 15.36pt 只是整体差异。
2. **专门做 target-contract ablation。** 如果 agent 在执行前显式生成 grain / columns / ordering / unit contract，能否显著降低 56.6% output-schema-related failures？
3. **给 join 加 invariants。** 显式 cardinality / uniqueness / row-count checks 能否稳定收回 9.7–19.8pt join gap？
4. **verification pass 是否值得独立模型预算。** 同一个模型 self-check、独立 verifier、deterministic checks 分别能减少多少 materialization failures？
5. **external state / lineage。** structured intermediate state 是否能在更低 token cost 下维持甚至提升 accuracy？
6. **真实 workspace benchmark。** 如果加入 stale files、version conflict、schema drift、missing values、权限边界和错误文档，当前 harness 排名会不会重排？
7. **DataSpace-Builder 开源程度。** 当前官方仓库未公开完整 builder 生成流水线；如果后续 release，值得单独审它如何防止 generation artifact、template cue 和 gold leakage。
8. **长期 contamination。** 410 个 input 全公开后，多久需要 hidden task rotation 或新版本 benchmark？
9. **table-only contract 的外延。** 能否保留 deterministic verification，同时扩展到 chart / structured report / analytical narrative 等更真实的 data-work 交付物？
