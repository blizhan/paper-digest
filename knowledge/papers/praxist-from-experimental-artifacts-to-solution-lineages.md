---
type: Research Paper Review
title: "Praxist: From Experimental Artifacts to Solution Lineages"
description: "Praxist 自主研究系统分析，重点拆解 typed Finding、task-defined Frontier、PI/Chair Agenda、peer-local design contract 与下一代 prompt 的 evidence inheritance 闭环，并讨论用 Aim/W&B 作为 evidence plane 复现其 research-control layer 的可行性。"
resource: https://arxiv.org/abs/2608.25955
tags: [autonomous-research, multi-agent, evidence, experiment-tracking, research-control-plane]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2608.25955
    title: "Praxist: From Experimental Artifacts to Solution Lineages"
  - id: code
    resource: https://github.com/sapientinc/praxist
    title: sapientinc/praxist
---

# Praxist: From Experimental Artifacts to Solution Lineages

## Links

- Paper: [arXiv:2608.25955](https://arxiv.org/abs/2608.25955)
- PDF: [arXiv PDF](https://arxiv.org/pdf/2608.25955)
- Official code: [sapientinc/praxist](https://github.com/sapientinc/praxist)
- Official docs: [Praxist documentation](https://praxist.sapient.inc/en/docs)
- 本次源码讨论主要基于官方仓库 commit `aa198d089fde2c6b9260b81af7cd0ac5965008a1`；整理笔记时（2026-09-02）官方仓库 HEAD 已前进到 `0d9598f85845cfba6ee6977423180b3c331b0371`，因此下面的实现描述记录的是我们实际审阅过的版本，而不是声称当前 HEAD 逐行不变。

## 一句话结论

Praxist 真正值得关注的不是“并行起很多 coding agents”，而是它在普通实验循环上增加了一层 **evidence → research policy → per-agent contract** 的控制层：实验先变成 typed `Finding`，再按成熟度与用途进入 task-defined `Frontier`，PI/Chair 把这些 evidence 转成下一代 `Agenda`，最终只把与当前 peer 相关的 contract、hypotheses 和少量 lineage/context 注入 prompt。

我们的核心判断是：**Praxist 最有研究价值的部分不是底层 experiment tracking，而是 research-policy compiler。** 因此 Aim / Weights & Biases 这类系统天然可以承担下面的 Evidence Plane；如果在它们之上补一个薄的 Finding / Frontier / Agenda controller，就能复现 Praxist 最关键的机制，并且比直接复制整套系统更适合做干净的 A/B ablation。

## 文章摘要

论文关注的是长期、可执行、可度量的 autonomous research。普通 coding-agent loop 往往把历史状态保存成聊天记录、solution tree、代码版本或按 score 排序的候选集合；随着 generation 增长，这些状态越来越难回答一个更重要的问题：**过去到底学到了什么，以及下一轮应该以什么身份继承这些 evidence？**

Praxist 的答案是把“实验产物”和“可继承研究知识”分开。并行 peers 构造并运行 artifact，task-owned evaluator 把结果投影成结构化 Finding；Finding 不只带 metric，还能带 hypothesis、parent、evidence maturity、failure mode、design dimensions 和 lineage relation。系统随后把 evidence 放进不同用途的 frontier lanes，再由 PI roles 与 Chair 做 generation-boundary synthesis，生成下一代 Agenda 和每个 peer 的 design/research contract。后续 peer 因而继承的是“一个已解释的研究任务”，而不是全量 transcript 或单一 leaderboard leader。

论文同时用 Quant、SLAM、MLE-bench/Jigsaw、rocket recovery 等 executable research campaigns 展示这种 lineage accumulation。最值得保留的信号不是“agent 又找到一个更高分模型”，而是系统能够显式保留：成熟 parent、脆弱 candidate、diagnostic failure、validation target、negative evidence，以及它们对下一代实验分工的影响。

## 方法拆解：真正的闭环是什么

可以把核心循环压缩成：

```text
experiment / evaluated artifact
          ↓
       Finding
          ↓
       Frontier
          ↓
      PI memos
          ↓
        Chair
          ↓
    Agenda_g+1
          ↓
 per-peer contract
          ↓
 prompt / experiment_g+1
          ↓
      Finding_g+1
```

这和普通 evolutionary search 的差别在于：**score 不是唯一的 inheritance type。** 一个高分 artifact 可以因为证据不成熟或 constraint violation 留在 incubator；一个低分 control 也可以作为 diagnostic evidence 继续影响后续研究。

## Finding 到底长什么样

当前公开实现里，peer 通过 `evaluation_tools.share_finding` 写结构化 Finding。核心字段包括：

```json
{
  "id": "uuid",
  "finding_type": "result",
  "title": "...",
  "content": "...",
  "metrics": {
    "task_metric": 0.0
  },
  "variant_name": "...",
  "peer_id": "gen15_peer10",
  "generation_id": 15,
  "design_dimensions": {
    "mechanism_family": "...",
    "intervention_surface": "..."
  },
  "links": [
    {
      "target_finding_id": "...",
      "edge_type": "derived_from",
      "rationale": "..."
    }
  ],
  "extra": {
    "peer_role": "exploit",
    "target_hypothesis": "H_g15_01",
    "evidence_stage": "...",
    "next_step_intent": "repair_failure_mode",
    "parent_candidate": "...",
    "parent_usage": "repair",
    "is_negative": false,
    "evidence_valence": "mixed",
    "failure_mode": "..."
  }
}
```

实现允许显式 Finding Graph edge：

```text
derived_from
updates
supports
challenges
related_to
```

所以 Finding 不是“`run.score = 0.91` 的另一种 JSON 包装”，而是在 score 之外增加了 **claim、provenance、research role 和 next-step semantics**。

### `planned_dimensions` 与 `design_dimensions` 的边界

Praxist 还特意把“计划的多样性位置”和“实际做出来的设计”分开：

- PI/Chair 在 peer contract 中写 `planned_dimensions`，表示 allocation intent；
- peer 完成实验后写 `design_dimensions`，表示真实 implementation/evidence；
- prompt 明确要求不能为了声称“按计划完成”而把 `planned_dimensions` 原样复制成 `design_dimensions`。

这一点很重要：QD 在这里管理的是 **research design allocation**，不是替最终 artifact 打标签。

## Frontier 不是 leaderboard

源码中的 canonical state 是 `frontier/frontier_manifest.json`。它至少维护：

```json
{
  "generations": {"15": []},
  "cumulative_top": [],
  "lane_frontiers": {},
  "validation_candidates": {
    "generations": {},
    "cumulative": []
  },
  "frontier_lanes": [],
  "primary_metric": "...",
  "metric_direction": "maximize"
}
```

被 promote 的 entry 不只是 `metric_value`，还可能携带：

- `finding_id` / `variant_name`；
- `frontier_lane` / `promoted_for_lane`；
- `parent_eligible`；
- compact metrics；
- evidence maturity / research metadata；
- `design_dimensions`；
- snapshot path 与 result identity/provenance。

论文用 `confirmed / candidate / diagnostic / validation` 解释方法，但**具体 lanes 是 task-defined，不是系统硬编码的四个枚举**。例如 Quant campaign 使用的是类似 `confirmed alpha / alpha incubator / benchmark floor / diagnostic control` 的 task-owned retention semantics。

因此更准确的理解是：

```text
Finding
  ↓
task-owned maturity / integrity / metric / lane policy
  ↓
它以什么“研究身份”被继承
```

而不是：

```text
Finding
  ↓
按 score 排序
  ↓
Top-K = 下一代 parent
```

## Agenda 才是下一代 Agent 的控制对象

`PIAgentConfig` 在源码里直接说明：PI 在 generation 之间读取上一代 state，写出 `research_agenda_gen{N+1}.yaml`，下一代 peers 会把其中的 role contracts 收到 prompt 中。

完整 Agenda 的代表性结构是：

```yaml
agenda_version: "2.0"
generation: 16
synthesized_from_gen: 15

mainline_observation:
  current_dominant_mechanisms: [...]
  main_risk: ...
  key_tradeoff: ...

cross_peer_hypotheses:
  - id: H_g16_01
    claim: ...
    source_findings: [...]
    minimal_test: ...
    kill_condition: ...
    promote_condition: ...

peer_contracts:
  gen16_peer3:
    role: exploit
    target_hypothesis: H_g16_01
    source_lane: ...
    target_lane: ...
    parent_candidate: ...
    parent_usage: repair
    next_step_intent: repair_failure_mode
    required_controls: [...]
    forbidden_actions: [...]
    success_signal: ...
    planned_dimensions: {...}

success_metrics:
  required: [...]
```

还可以带：

- `bridge_hypothesis`；
- `anti_mainline_contract`；
- `falsification_contract`；
- dissent / claim-boundary / consensus actions；
- generation-level success metrics。

这说明 Chair 的工作不是写一句“继续优化最优方案”，而是在做 **portfolio-level research allocation**：谁 exploit、谁 repair、谁 falsify、谁 bridge、谁必须避开主线。

## 下一代 prompt 不是塞全量历史，而是 peer-local slice

我们读源码时最关键的发现之一，是 `prompt_context.py` 会先对完整 Agenda 做压缩，只给当前 peer 保留：

```text
完整 Agenda
   │
   ├── 当前 peer 自己的 contract        → 完整保留
   ├── 与它相关的 hypothesis            → 保留
   ├── current_peer_source_context       → 保留
   ├── 少量相关 sibling roster          → 保留
   ├── generation-level success metrics → 保留
   └── 大量无关 cohort state            → 省略/压缩
```

`prompt_generation.jinja2` 随后把 contract 直接渲染进 prompt，包含：

- role；
- target hypothesis；
- source / target lane intent；
- required controls；
- forbidden actions；
- success signal；
- `planned_dimensions`；
- relevant source context；
- cross-peer hypotheses / bridge / falsification 信息（仅在相关时）。

模板还明确写出：**role contract 对 broader task prompt 的默认指令有覆盖优先级。** 因此它不是“memory hint”，而是真正的下一代 design contract。

而 peer 结束后又被要求把 `peer_role`、`target_hypothesis`、`parent_candidate`、`parent_usage`、`next_step_intent`、negative-evidence metadata 等写回 Finding，于是形成可审计闭环。

## Quant case study：最清楚的一条 evidence → repair 链

论文的 Quant 案例非常适合说明为什么不能只看 leaderboard。

### 轨迹

论文描述的核心 lineage 是：

```text
较早的 LSTM-PPO recurrent policy
        │
        │  return 很强
        ▼
frontier 保留 recurrent-PPO parent
        │
        └── diagnostic finding:
              concentration risk 太高
                    │
                    ▼
             diversification repair
                    │
          effective-number regularization
          max-weight penalties / constraints
                    │
                    ▼
   gen15_peer10_diversify_repair_strong
```

reported policy 的关键结果：

- 2019Q1–2025Q4 28 个 quarterly windows 上，calendar-time CAGR **53.07%**；
- paired equal-weight baseline CAGR **22.80%**；
- 2026 training-excluded validation return **21.85%**，max drawdown **10.97%**，daily Sharpe **1.73**；
- repair 后 mean effective names **8.01**；
- maximum mean single-name weight **21.69%**；
- mean daily turnover **7.95%**；
- mean cash **9.57%**。

但 preregistered diversification success signal 要求：

```text
effective names > 10
max single-name weight < 15%
```

它没有达到，而且论文明确说明该 artifact 只有 T1 单 seed / 29 cells evidence，并携带 **3 个 hard constraint violations**。所以虽然它是 campaign 中记录到的最高 walk-forward CAGR artifact，却仍然只在 **incubator lane**，不是系统 clean-promotion 后的 champion。真正 strongest confirmed-lane result 是另一个 generation-19 cross-sectional attention policy，它完成了五 seeds / 145 cells 的 T3 clean evaluation，但 CAGR 更低。

这条链把 Praxist 的核心价值解释得很清楚：

```text
53.07% CAGR
   ↓
高价值 evidence
   ↓
但 maturity / constraints 不合格
   ↓
incubator / repair target
   ↓
不能自动成为 confirmed parent
```

### 明确事实与我们的映射

论文明确写了“recurrent-PPO parent → concentration diagnostic → diversification repair → repaired artifact”的研究轨迹，也明确给出上述指标和 lane/status。

但公开材料没有发布这个 Quant campaign 当时真实的 `research_agenda_gen*.yaml`。因此把这条轨迹对应到当前 schema 的：

```yaml
parent_usage: repair
next_step_intent: repair_failure_mode
source_lane: alpha_incubator
target_lane: confirmed_alpha
```

是**基于当前公开实现的结构化映射**，不是当年某个 Agenda 文件的逐字转录。`gen15_peer10_diversify_repair_strong` 这个名字强烈指向 generation 15 / peer 10，但我们也不把这个命名推断扩大成“已经拿到 gen15 原始 prompt”。

## SLAM case study：机制很清楚，但逐 generation contract 没公开

SLAM 案例的最终机制是 **COVSCHED**，在 FAST-LIVO2 的 visual pathway 上增加两类控制：

1. `PVTR_MODE=8`：基于 LiDAR translation observability 的 visual-update scheduler；
2. `VMAP_DEDUP=1`：visual-map admission filter，对已经被 map 表达的几何重观察进行去重拒绝。

论文报告：

- across 14 sequences，evaluator-captured VIO processing time 平均下降 **72.4%**，median **74.4%**；
- 将内部 LIO + VIO thread-wall workload 相加作为保守 aggregate proxy，下降 **22.1%**；
- 在 `nya_03` 上，高 observability frames skip rate **74.9%**，低 observability frames 只有 **0.3%**；
- map admission filter 使用约 **0.08m** 距离与 **15°** normal-deviation 条件避免重复点增长。

但这篇在这里也展示了很重要的 evidence boundary：raw APE 看起来大幅变好，却存在 pose timestamp confound。COVSCHED 与 baseline 的 timestamp convention 不同，而 evaluator 的 pose association 对这个差异敏感；用一致规则重新关联后，mean relative APE change 只剩约 **-0.09%**。所以论文最终只支持“显著减少 visual computation 而不明显牺牲 trajectory accuracy”，不再把 raw APE gap 声称成算法 accuracy gain。

这个 case 很能说明为什么 typed evidence / claim boundary 有价值：**强结果不等于强 claim，control/measurement confound 也应该成为后续 research policy 的一部分。**

但论文没有像 Jigsaw / Rocket 部分那样公开一条 COVSCHED 的逐 generation lineage，官方 GitHub 当前也没有附带当时 SLAM campaign 的 `research_agenda_gen*.yaml`。所以我们不能声称知道“某一代 Chair 原文如何要求下一代 peer 修改 scheduler”。

## Jigsaw：论文里更显式的 lineage 示例

Jigsaw Toxic Comment Classification 的 trajectory 更适合观察 lineage accumulation：

```text
Gen 0 class-balanced focal-loss BERT       0.98620
        ↓ matched BCE ablation
Gen 0 BCE parent                          0.98657
        ↓
Gen 1 two-BERT average                    0.98679
        ↓ + DistilRoBERTa
Gen 1 three-model parent                  0.98723
        ├── + BiLSTM                      0.98711  negative finding
        └── + RoBERTa-CLS                 0.98749  confirmed selection
```

同一代还保留 metadata-only null model（0.76106）作为 diagnostic finding。论文说明最终 finding 记录了指向 three-model result 的 `updates` link 和指向 two-seed BERT ensemble 的 `derived_from` link。

这里最重要的不是 0.000x 的分数提升，而是**成功分支、失败分支、null control 和 parent chain 都没有被压成一个最终 checkpoint**。

## DIG / QD：控制“研究设计多样性”，不是只做 candidate diversity

Praxist 的 Deep Innovation Gate（DIG）在默认配置下主要用于 opening allocation。它要求 peer 在写代码前先固定：

- mechanism family；
- intervention surface；
- parent lineage；
- evidence signature；
- validation / ablation hook；
- forbidden changes。

QD 再按类似：

```text
(mechanism family, intervention surface, intent)
```

的 design-cell coordinates 分配 peers，避免 cohort 全部追同一条高分方向。当前实现里 opening DIG 可以是硬 allocation；后续 generation 则更多由 Chair 通过 `planned_dimensions` 做 soft diversity planning。

这和经典 QD 的相似点是“维持 feature-space coverage”，不同点是 Praxist 把 feature space 用在 **research proposal / intervention design**，而不是只用来保留最终 artifact。

## GitHub / Code Analysis

我们实际沿着下面的源码路径确认了 Finding → Frontier → Agenda → prompt 的写读链：

- `praxist/plugins/tools/evaluation_tools/adapter.py`
  - `share_finding` 接收 metrics、links、`design_dimensions`、`extra`；
  - `extra.peer_role` 与 `extra.target_hypothesis` 会保留下来供 frontier / PI synthesis 使用；
  - Finding 会写入共享 filesystem / store，并可即时进入 Finding Graph rule engine。
- `praxist/plugins/workflow_stages/research_loop/backend/frontier.py`
  - `FrontierStore` 维护 `frontier_manifest.json`；
  - 支持 primary metric、anchor metrics、task-defined frontier lanes、validation candidates、maturity/risk gating；
  - promoted entry 会保留 finding identity、metrics、lane、research/evidence metadata 和 snapshot。
- `praxist/plugins/workflow_stages/research_loop/backend/pi_agent.py`
  - generation boundary 后写 `research_agenda_gen{N+1}.yaml`；
  - validator 要求 `peer_contracts` 结构、canonical peer IDs、role、target hypothesis、success signal 等。
- `praxist/plugins/workflow_stages/research_loop/backend/multi_pi/chair_arbiter.py`
  - Chair 把 PI memos、objections、hypotheses 和 frontier context 组合成 final agenda；
  - fallback path 也会保留 bridge、anti-mainline、falsifier、required controls 与 per-peer contracts。
- `praxist/plugins/workflow_stages/research_loop/backend/prompt_context.py`
  - `_compact_research_agenda_for_prompt` 只保留当前 peer contract 和相关 context；
  - sibling contracts 只保留 compact roster / summary，避免把整个 cohort 的详细计划广播给每个 peer。
- `praxist/plugins/workflow_stages/research_loop/backend/prompt_generation.jinja2`
  - 真正把 `role / target_hypothesis / lane intent / required_controls / forbidden_actions / success_signal / planned_dimensions` 渲染进下一代 prompt；
  - 同时要求 peer 把 provenance 和 negative-evidence metadata 写回 Finding。

这条调用链是我们认为整篇最值得保留的实现事实：**Agenda 不只是报告，而是 prompt compiler 的输入。**

## 我们的核心洞察：Aim / W&B 可以天然承接 Evidence Plane

讨论到最后，一个很自然的问题是：Praxist 是否真的需要自己拥有这么重的一整套 evidence ingestion/storage？

我们的判断是：**不需要。Aim / Weights & Biases 已经能天然承担很大一部分“可验证实验世界”；Praxist 真正需要复现的是其上的 research-control layer。**

可以做如下映射：

| Praxist object | Aim / W&B primitive |
| --- | --- |
| evaluated artifact | Run + Artifact / checkpoint / result file |
| metrics | run metrics / summary |
| config / intervention | config / parameters / tags |
| code / execution provenance | run metadata + source/version metadata |
| Finding | typed metadata，或一个独立 Finding JSON artifact/table row |
| `design_dimensions` | config / custom metadata / artifact metadata |
| lineage | artifact lineage + 自定义 relation metadata |
| Frontier | 对 runs/findings 查询后 materialize 的 controller-owned state |
| confirmed / incubator / diagnostic / validation | controller policy + collection/tag/materialized view |
| Agenda | controller 生成的 YAML/JSON artifact |
| peer contract | Agenda 中每个 agent 的 slice |
| next-generation prompt | orchestrator 根据 Agenda + relevant evidence 编译 |
| Gems | 周期性生成的 compressed durable knowledge artifact |

### 哪些东西 experiment tracker 已经免费提供

Aim/W&B 类系统通常已经解决：

- run identity；
- metric history / summary；
- hyperparameter/config；
- logs；
- artifact/checkpoint 管理；
- code/version provenance；
- dashboard / query；
- 不同程度的 artifact lineage。

这些恰好是 Praxist 最底层 evidence plane 最难维护、但又不是论文最独特的部分。

### 哪些东西仍然必须自己补

Aim/W&B 默认不会替你判断：

```text
这个 53% CAGR 虽然最高，
但证据只到 T1 且有 hard violations，
所以只能进 incubator，不能作为 confirmed parent。

这个失败实验不是垃圾，
它是 diagnostic evidence。

下一轮 peer3 不继续刷 CAGR，
而是继承这个 parent，专门 repair concentration，
并且 success signal 是 max_weight < 0.15。
```

也就是说需要新增的是：

```text
Evidence
   ↓
typed Finding
   ↓
operational status / Frontier
   ↓
research policy / Agenda
   ↓
per-agent contract
```

这才是 Praxist 的核心差异层。

## 最小可行的 W&B/Aim Research Controller

如果我们自己复现，不建议一开始复制整个 Praxist。最小架构可以只有五层：

```text
             Aim / W&B
        ┌─────────────────┐
        │ Runs            │
        │ Metrics         │
        │ Config          │
        │ Artifacts       │
        │ Logs / lineage  │
        └────────┬────────┘
                 │ normalize
                 ▼
          Finding Extractor
                 │
                 ▼
          Typed Findings
                 │
        ┌────────┴────────┐
        ▼                 ▼
 Frontier Engine       PI / Chair
        │                 │
        └───────┬─────────┘
                ▼
          agenda_genN.yaml
                │
                ▼
        peer-local compiler
                │
                ▼
              Agents
                │
                ▼
             Aim / W&B
```

最小实现职责：

1. **tracker adapter**：把 Aim/W&B Run 统一投影成 `Evidence`；
2. **Finding schema**：增加 hypothesis / parent / maturity / valence / failure mode / design dimensions；
3. **Frontier evaluator**：纯函数决定 confirmed / incubator / diagnostic / validation 等 operational status；
4. **PI/Chair**：LLM 只读 compact findings + frontier，输出 schema-validated `agenda.yaml`；
5. **prompt compiler**：只把当前 peer contract + referenced evidence 注入上下文。

这样已经复现：

```text
W&B/Aim Run
 → Finding
 → Frontier
 → PI/Chair
 → Agenda
 → peer contract
 → new Run
```

而不用同时复制 Praxist 的 lifecycle、scheduler、credential、resume、plugin、Gems 等全部基础设施。

## 新洞察：NotebookLM 可以成为 Literature Frontier

进一步讨论后，我们认为 PI/Chair 不应该只消费本 campaign 的实验 Findings。它还可以接一条独立的 **Literature Plane**：用 NotebookLM 对前沿论文、方法综述、项目文档和已有研究笔记做 grounded extraction，把外部知识整理成可执行的 `Literature Finding`，再交给 PI/Chair 形成下一代 hypothesis / contract。

关键不是让 NotebookLM 只回答“有哪些相关论文”，而是要求它把文献压成接近 Finding 的结构化对象：

```yaml
literature_finding:
  claim: "Observability-gated visual updates can reduce redundant computation"
  mechanism:
    family: adaptive_sensor_scheduling
    intervention_surface: visual_update_frequency
  evidence:
    supporting_papers: [...]
    conflicting_papers: [...]
    evidence_strength: medium
  applicability:
    conditions: [...]
    risks: [...]
  proposed_test:
    parent_candidate: current_best_slam
    intervention: add_observability_gated_scheduler
    minimal_ablation: [...]
  falsification: [...]
```

也就是说：

```text
paper / docs / prior work
        ↓
     NotebookLM
        ↓
Literature Findings / Idea Cards
        ↓
 Literature Frontier
```

### 最有价值的用法：让当前 Frontier 的问题反向驱动文献检索

相比定期泛读论文，更有价值的是由 Experimental Frontier 主动生成 NotebookLM query。

例如 Quant 当前状态是：

```text
best recurrent PPO
CAGR 很高

diagnostic:
  concentration too high
  effective names = 8.01
  target > 10
```

PI 可以把这个 failure mode 转成 literature query：寻找能保持 alpha、但改善 diversification 的 recurrent-RL-compatible mechanism，并要求 NotebookLM 同时给出已知 failure modes、minimal ablation 和 falsification condition。

NotebookLM 返回的 effective-number regularization、entropy constraint、differentiable top-k、risk-budget penalty 等方向，不直接成为“已验证事实”，而是成为下一代可分配的 research-action candidates。Portfolio PI / Chair 再据此把不同机制分给不同 peers，天然和 QD 的 mechanism-family coverage 对接。

### 必须保持三种 Frontier 的语义隔离

这里最重要的边界是：**NotebookLM 提取出的论文 evidence 不能和自己实验产生的 evidence 放进同一个 Frontier 直接排名。**

更合理的状态分层是：

```text
Experimental Frontier
    = 我们在当前 task / evaluator 下实际测出来了什么

Literature Frontier
    = 外部论文和文档报告了什么、哪些机制可能迁移

Idea / Hypothesis Frontier
    = 尚未被当前 task 验证、但值得分配实验的方向
```

正确转换应当是：

```text
Literature Finding
       ↓
PI 判断 task applicability
       ↓
hypothesis / idea
       ↓
peer contract
       ↓
our experiment
       ↓
Experimental Finding
       ↓
Experimental Frontier
```

这样可以避免 `paper says X works` 被错误升级成 `X is a confirmed parent in our task`，也保持 canonical experimental truth 只由当前 evaluator 的真实测量产生。

### 合并后的完整架构

这使我们前面的 Aim/W&B 方案进一步变成一个双证据面架构：

```text
Aim / W&B                         NotebookLM
   │                                  │
   ▼                                  ▼
Experiment Evidence              Literature Evidence
   │                                  │
   ▼                                  ▼
Experimental Frontier            Literature Frontier
            \                      /
             \                    /
              └──────→ PI / Chair
                         │
                         ▼
                 Idea / Hypothesis Frontier
                         │
                         ▼
                   Agenda Compiler
                         │
                         ▼
                   Agent Contracts
                         │
                         ▼
                     Experiments
                         │
                         └────────→ Aim / W&B
```

可以把三类职责压缩成一句话：

> **NotebookLM 提供 grounded scientific prior；Aim/W&B 提供 grounded empirical evidence；PI/Chair 把两类 evidence 编译成下一步 research policy。**

这已经不只是“在 Aim/W&B 上复现 Praxist”，而是一个更清楚、也更容易单独验证的 research-agent architecture。

## 为什么这种复现反而更适合验证论文核心 claim

把底层 tracker 固定以后，可以做非常干净的 A/B：

```text
A: Aim/W&B + 普通 autonomous coding/research agent
B: Aim/W&B + Finding/Frontier/Agenda research controller
```

保持 evaluator、模型、agent runtime、budget、experiment storage 都相同，只改变 evidence inheritance / research-control layer，然后比较：

- 最终 task metric；
- 达到某 performance threshold 的实验数 / wall time / compute；
- 重复或近重复实验率；
- invalid / non-promotable experiments 比例；
- lineage depth / useful recombination rate；
- constraint violations；
- negative evidence 被后续使用的比例；
- design diversity / mechanism-family coverage；
- 每代 prompt token cost 与 context relevance。

这会比“Praxist 全系统 vs 另一个全系统”更容易回答真正的问题：

> **把 raw experiment history 变成 typed evidence，再编译成下一代 research contract，到底有没有独立贡献？**

## 关键分析

### 1. Praxist 不是高级 experiment memory

如果只把它理解成“长期记忆 + leaderboard”，会错过最关键的一层。它真正新增的是：

```text
memory/evidence
   ↓
operational interpretation
   ↓
future-work allocation
```

即一个 research-policy compiler。

### 2. Frontier lane 的价值在于把“好”拆成不同语义

研究里的“好”至少有：

- performance 好；
- evidence 成熟；
- 适合作为 parent；
- 值得 validation；
- diagnostic 信息量高；
- negative result 足以排除一个方向。

单 metric leaderboard 会把这些角色压成一个 scalar ranking。Praxist 的 lane design 是对这个问题的直接回答。

### 3. 下一代 prompt 的关键不是更多 context，而是更强的 selection

Praxist 没有把“所有历史都喂给下一代”当作默认答案，而是把 Agenda 编译成 peer-local slice。这和我们之前在 Data Agent / long-context 系统里得到的判断一致：**structured state + targeted retrieval 往往比扩大 transcript context 更重要。**

### 4. Negative evidence 只有在能改变 allocation 时才真正有价值

“记录失败实验”本身不新。真正新的是失败可以进入 diagnostic lane、成为 falsification/repair contract 的 source，并改变下一代谁做什么。

### 5. QD 最值得借的是 allocation semantics，不一定要照搬 DIG

即使不用 Praxist 的 DIG 实现，也可以在 controller 中要求每个 Agenda 明确计划：mechanism family、intervention surface、intent、parent lineage，并对 cohort 做覆盖约束。这样已经能保留 QD 的主要研究分工价值。

## 我们的观点

Praxist 的工程体量很大，但从研究创新角度看，最值得继续追的其实可以压缩成一句话：

> **把 experiment tracker 中的 run/artifact 解释成 typed research evidence，再把 evidence 编译成下一代 agent contract。**

这比“多 agent + 多 generation”更有辨识度，也更容易迁移到其他平台。

同时，论文目前最缺的也是这一层的干净 ablation。Quant / SLAM case studies 证明系统可以跑出有意思的长程 lineage，源码也证明 Finding → Frontier → Agenda → prompt 确实在执行；但还没有非常强地回答：

```text
如果底层 agent、evaluator、budget 完全不变，
去掉 typed frontier / PI-Chair contract 后到底损失多少？
```

因此我们不把 case-study 成果直接等价成“research-policy layer 已被因果验证”。这反而是用 Aim/W&B 做薄层复现最值得验证的方向。

## 值得继续追的问题

1. **核心 ablation**：`raw run history` vs `typed Finding` vs `Finding+Frontier` vs `Finding+Frontier+Agenda` 各自贡献多少？
2. **PI/Chair necessity**：固定 rule-based controller 能否达到大部分收益，LLM synthesis 真正增加多少？
3. **lane design sensitivity**：task-defined lanes 的收益来自语义本身，还是来自更保守的 promotion gate？
4. **negative evidence reuse**：多少 diagnostic / negative findings 真正改变了未来 contracts，而不是只被记录？
5. **prompt slicing**：peer-local agenda 相比完整 agenda / full transcript 在 token cost、重复工作率和最终结果上差多少？
6. **QD allocation**：只做 `planned_dimensions` coverage，是否已经足以替代复杂 DIG？
7. **tracker-backed reproduction**：Aim/W&B 上的最小 controller 是否可以在相同 evaluator 和 agent runtime 下复现类似的 lineage depth / research efficiency？
8. **case-study artifact release**：如果未来官方公开 Quant/SLAM 的原始 `research_agenda_gen*.yaml`、rendered prompts 和 full lineage ledger，可以进一步核对论文叙述与真实 generation-by-generation control path。
9. **literature-conditioned control**：加入 NotebookLM Literature Frontier 后，是否能提升 novel mechanism discovery、降低重复踩坑，并提高下一代 contracts 的可解释性，而不污染 experimental canonical state？
