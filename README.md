# Papers

这个仓库用来记录我们**已经实际读过、讨论过**的论文。重点不是做论文收藏夹，而是留下后续还能复用的判断：论文到底解决了什么、核心方法是否真的新、实验说明了什么、有哪些值得继续追的 insight，以及公开代码实际实现到了哪一步。

知识内容按 [Open Knowledge Format (OKF) v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) 的思路组织在 [`knowledge/`](./knowledge/index.md)：目录通过 `index.md` 渐进式暴露内容，每篇论文是一份带 YAML frontmatter 的独立 concept 文档。根 `README.md` 保留为面向人的仓库入口。

## 知识入口

- [OKF bundle index](./knowledge/index.md)
- [论文索引](./knowledge/papers/index.md)

## 当前论文

### ICL

| Paper | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [Causilo Technical Report](./knowledge/papers/causilo-technical-report.md) | tabular foundation model, ICL, efficient attention, probabilistic regression | [arXiv](https://arxiv.org/abs/2609.22866) · [GitHub](https://github.com/nums-ai/causilo) · [Weights](https://huggingface.co/nums-ai/causilo) | 已讨论 / 已读源码 | 先行内细化、再二次读取 context、最后压缩行表示，取得强的准确率/速度折中；合成数据与模块贡献尚无法独立归因。 |
| [Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI](./knowledge/papers/advancing-open-and-reproducible-relational-learning-relarena-tabpfn-rel-rpi.md) | relational learning, tabular foundation model, benchmark | [arXiv](https://arxiv.org/abs/2608.16319) · [Hugging Face](https://huggingface.co/papers/2608.16319) · [GitHub](https://github.com/PriorLabs/relarena) | 已讨论 | 核心贡献更偏 benchmark / reproducibility / relational pipeline；TabPFN-Rel 的关键结论是“DFS flattening + 强 tabular backbone”依然能和专门的 relational architecture 正面竞争。 |
| [GraphPFN: A Prior-Data Fitted Graph Foundation Model](./knowledge/papers/graphpfn-a-prior-data-fitted-graph-foundation-model.md) | graph foundation model, PFN, ICL, synthetic graph prior | [arXiv](https://arxiv.org/abs/2509.21489) · [Hugging Face](https://huggingface.co/papers/2509.21489) · [GitHub](https://github.com/yandex-research/graphpfn) | 已讨论 | 真正贡献是证明“LimiX 初始化 + graph adapter + graph-aware synthetic pretraining”可以强力迁移到真实 node-level tasks；复杂 multi-level SBM + PA prior 的必要性则未被自身 ablation 证明。 |
| [LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence](./knowledge/papers/limix-2-a-contextual-mechanism-network-towards-general-structured-data-intelligence.md) | tabular foundation model, CCMM, ICL, causal skeleton | [arXiv](https://arxiv.org/abs/2609.17488) · [GitHub](https://github.com/limix-ldm/LimiX) · [Weights](https://huggingface.co/stableai-org/LimiX-2) | 已讨论 / 已读源码 | 沿用 CCMM 的多变量条件推断与非对称双轴 Attention；因果发现实验仅恢复无向骨架，公开 v2 推理代码尚不能直接导出所需 Attention 权重与复现骨架评测。 |
| [Mitra-v2 Technical Report](./knowledge/papers/mitra-v2-technical-report.md) | tabular foundation model, synthetic prior, Hybrid SCM, FT / bagging | [arXiv](https://arxiv.org/abs/2609.04540) · [Code / Results](https://huggingface.co/autogluon/mitra-finetune) · [Weights](https://huggingface.co/autogluon/mitra-classifier-2) | 已讨论 / 已读源码 | 扩大合成任务分布且保留 Tab2D 主干；论文主成绩依赖 FT + 8-fold bagging 与 support/宽表部署配方，Hybrid SCM 单项贡献尚未隔离。 |
| [Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality (O'Prior)](./knowledge/papers/shaping-the-prior-how-synthetic-task-distributions-determine-tabular-foundation-model-quality.md) | tabular foundation model, synthetic prior, Hybrid SCM, distribution shift | [arXiv](https://arxiv.org/abs/2605.18971) · [HTML](https://arxiv.org/html/2605.18971v1) | 已讨论 / 已读原文 | 将结构生成、观测真实性、shift stress 与课程/防泄漏约束分开设计和消融；Hybrid SCM 收益明显，但完整配方在固定短预算下不总占优。 |
| [TabICLv2: A better, faster, scalable, and open tabular foundation model](./knowledge/papers/tabiclv2-a-better-faster-scalable-and-open-tabular-foundation-model.md) | tabular foundation model, ICL, synthetic prior, domain adaptation | [arXiv](https://arxiv.org/abs/2602.11139) · [GitHub](https://github.com/soda-inria/tabicl) · [NanoTabICL](https://github.com/soda-inria/nanotabicl) | 已讨论 | 最值得保留的是高组合度 GraphSCM prior 与 dataset-level ICL 训练范式；它提供很强的 structural prior，但仍基本缺失 domain semantics，因此自然引出 continued PT、semantic encoder 与 domain-conditioned prior 三条私有领域适配路线。 |
| [Xiaomi-TabLDM: A Tabular Foundation Model Technical Report](./knowledge/papers/xiaomi-tabldm-a-tabular-foundation-model-technical-report.md) | tabular foundation model, SCM prior, ICL, quantile regression | [arXiv](https://arxiv.org/abs/2609.03880) · [GitHub](https://github.com/xiaomi-research/xiaomi-tabldm) · [Weights](https://huggingface.co/occams/Xiaomi-TabLDM) | 已讨论 / 已读源码 | 双流特征分组、AttnRes 与 MoE 扩展 TabICLv2 式 ICL 架构；基础模型可输出 999 个分位数，普通多视图逐分位数平均，NNLS 增强模式目前仅返回点预测，概率校准仍待验证。 |

### SSL

| Paper | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [SwitchTab: Switched Autoencoders Are Effective Tabular Learners](./knowledge/papers/switchtab-switched-autoencoders-effective-tabular-learners.md) | tabular learning, SSL, representation learning, feature engineering | [arXiv](https://arxiv.org/abs/2401.02013) · [Hugging Face](https://huggingface.co/papers/2401.02013) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29523) · [Unofficial GitHub](https://github.com/avivnur/SwitchTab) · [TabularS3L](https://github.com/Alcoholrithm/TabularS3L) | 已讨论 | Switching constraint 本身有一致 ablation signal，但 `s/m` disentanglement 证据弱；纯 SSL 适合作为自动特征候选而非已证明的最优 extractor。两份第三方实现均已审计，TabularS3L 更完整但仍有 reconstruction target 与 feature-export bug。 |

### Ensemble / Efficient DL

| Paper | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](./knowledge/papers/tabm-advancing-tabular-deep-learning-with-parameter-efficient-ensembling.md) | tabular DL, parameter-efficient ensemble, BatchEnsemble | [arXiv](https://arxiv.org/abs/2410.24210) · [GitHub](https://github.com/yandex-research/tabm) | 已讨论 | 把 ensemble member 变成显式并行维度，并通过大权重共享、member-specific modulation/head 和 ensemble-aware stopping 获得强平均预测；最值得追的是 weak-individual / strong-ensemble 机制，而不是单 member 变强。 |
| [TabPack: Efficient Hyperparameter Ensembles for Tabular Deep Learning](./knowledge/papers/tabpack-efficient-hyperparameter-ensembles-for-tabular-deep-learning.md) | packed ensemble, HPO, population training, AutoML | [arXiv](https://arxiv.org/abs/2607.05380) · [GitHub](https://github.com/yandex-research/tabpack) | 已讨论 | 把传统 HPO 改写为一次 packed heterogeneous population training + online checkpoint/ensemble selection；异构 HP 的现有证据主要支持降低 tuning 成本，ResNet/CNN 与固定图 GCN 是很自然的下一步推广对象。 |
| [Mapping Networks](./knowledge/papers/mapping-networks.md) | meta-parameterization, hypernetwork, low-dimensional weight manifold, parameter-efficient training | [arXiv](https://arxiv.org/abs/2602.19134) · [CVPR](https://openaccess.thecvf.com/content/CVPR2026/papers/Sen_Mapping_Networks_CVPR_2026_paper.pdf) · [Poster](https://cvpr.thecvf.com/virtual/2026/poster/36440) | 已讨论 | 真正减少的是 optimization DOF / trainable parameters，而不是总存储和 FLOPs；作为低维 model latent space 的入口很有意思，尤其自然引出 dataset→z→model、latent Bayesian ensemble、model diffusion 和 WebGPU 小码分发。 |

### Time Series

| Paper / Release | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [Chronos-2: From Univariate to Universal Forecasting](./knowledge/papers/chronos-2-from-univariate-to-universal-forecasting.md) | time-series foundation model, multivariate forecasting, covariates, ICL, synthetic data | [arXiv](https://arxiv.org/abs/2510.15821) · [GitHub](https://github.com/amazon-science/chronos-forecasting) · [Hugging Face](https://huggingface.co/amazon/chronos-2) | 已讨论 | 真正有价值的是用 group attention + synthetic multivariate prior 把 univariate / multivariate / covariates 统一成 zero-shot ICL；实验上纯 multivariate cross-target 增益很小，而 covariates 带来明显提升。 |
| [Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](./knowledge/papers/learning-recursive-multi-scale-representations-for-irregular-multivariate-time-series-forecasting.md) | irregular time series, multi-scale forecasting, representation fusion | [arXiv](https://arxiv.org/abs/2602.21498) · [OpenReview](https://openreview.net/forum?id=JEIDxiTWzB) · [GitHub](https://github.com/Ladbaby/PyOmniTS) | 已讨论 | ReIMTS 不靠 resampling 构造尺度，而是按真实时间 period 做 top-down recursive split；真正亮点是保留 sampling pattern，并用 padding/mask + batch reshape + gated residual 解决不等长跨尺度融合。 |
| [TimesFM-3: A zero-shot foundation model for multivariate forecasting](./knowledge/papers/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting.md) | time-series foundation model, multivariate forecasting, covariates, zero-shot | [Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · [GitHub](https://github.com/google-research/timesfm) · [Hugging Face](https://huggingface.co/google/timesfm-3.0-pytorch) | 已讨论 | 实质升级是把 multiple targets、past-only / past-future covariates 和 cross-variate attention 原生放进模型，并用 CPM 做 single-pass horizon；不是第一个支持 future covariates，但当前公开 benchmark 属于最强一档，3.0 权重目前不可商用。 |

### Weather / Forecasting

| Paper | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [WeatherNext 3: Increasing resolution and performance of global weather models with raw observations](./knowledge/papers/weathernext-3-increasing-resolution-and-performance-of-global-weather-models-with-raw-observations.md) | global weather forecasting, probabilistic ensemble, satellite observations, continuous decoding | [arXiv](https://arxiv.org/abs/2609.03582) · [Project](https://deepmind.google/science/weathernext/) · [GitHub](https://github.com/google-deepmind/weathernext) | 已讨论 | 真正变化是把 analysis、低延迟卫星、降水/站点 observation 和 cyclone targets 统一进一个 global probabilistic model，并通过 hourly refresh 与 continuous station head 改变 forecast product interface；但它仍是 analysis-anchored，而不是完整替代 data assimilation 的 raw-observation forecaster。 |

### Agent

| Paper | Topic | Links | Status | 一句话判断 |
| --- | --- | --- | --- | --- |
| [DataSpace: Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces](./knowledge/papers/dataspace-benchmarking-data-agents.md) | data agent, heterogeneous workspace, benchmark, verifiable analytics | [arXiv](https://arxiv.org/abs/2608.03451) · [Hugging Face](https://huggingface.co/papers/2608.03451) · [Project](https://dataspace-bench.github.io/) · [GitHub](https://github.com/HKUSTDial/DataSpace) | 已讨论 | 真正有价值的是把 Data Agent 定义成“在异构 workspace 中发现证据、完成关系计算并可靠 materialize 最终表”的系统问题；15.36pt harness gap 和 52.2% materialization failures 都说明 execution / verification / context management 不能当配角。 |
| [Praxist: From Experimental Artifacts to Solution Lineages](./knowledge/papers/praxist-from-experimental-artifacts-to-solution-lineages.md) | autonomous research, multi-agent, evidence inheritance, research control plane | [arXiv](https://arxiv.org/abs/2608.25955) · [GitHub](https://github.com/sapientinc/praxist) | 已讨论 | 真正有辨识度的不是多 agent，而是把 experiment 变成 typed Finding，经 task-defined Frontier 和 PI/Chair Agenda 编译成下一代 peer-local design contract；Aim/W&B 很适合作为底层 evidence plane，值得单独 ablate 这层 research-policy compiler。 |

后续只有当新论文形成明显的新方向时，再新增新的一级类别；暂时不为了分类完整性预先创建空类别。

## 记录约定

每篇论文作为 `knowledge/papers/` 下的一份独立 Markdown concept，至少记录：

- 文章摘要与问题定义；
- 关键方法与实验结论；
- 对方法真正贡献点的拆解；
- 值得保留的 insight；
- 我们自己的判断，明确和论文事实分开；
- 如果有公开代码，分析 GitHub 的实现路径、工程结构、可复现性与实际限制。

暂时不把“只是看到、但还没有实际读到/讨论到”的论文加进来。
