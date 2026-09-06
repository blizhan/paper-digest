---
type: Technical Article Review
title: "TimesFM-3: A zero-shot foundation model for multivariate forecasting"
description: "TimesFM-3 技术发布与源码分析：原生 multivariate / past-future covariates、时间轴 + 变量轴交替 attention、CPM single-pass decoding，以及与 Chronos-2、Moirai、PFN/ICL 路线的公开 benchmark 对比。"
resource: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
tags: [time-series-foundation-model, forecasting, multivariate, covariates, zero-shot, transformer]
status: stable
sources:
  - id: article
    resource: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
    title: "TimesFM-3: A zero-shot foundation model for multivariate forecasting"
  - id: code
    resource: https://github.com/google-research/timesfm
    title: google-research/timesfm
  - id: weights
    resource: https://huggingface.co/google/timesfm-3.0-pytorch
    title: google/timesfm-3.0-pytorch
  - id: cpm
    resource: https://arxiv.org/abs/2505.23719
    title: Contiguous Patch Masking
  - id: gift-eval
    resource: https://huggingface.co/spaces/Salesforce/GIFT-Eval
    title: Salesforce GIFT-Eval
  - id: fev-bench
    resource: https://huggingface.co/spaces/autogluon/fev-bench
    title: AutoGluon FEV-Bench
---

# TimesFM-3: A zero-shot foundation model for multivariate forecasting

## Links

- Google Research article: [TimesFM-3: A zero-shot foundation model for multivariate forecasting](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- Official code: [google-research/timesfm](https://github.com/google-research/timesfm)
- Official weights: [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)
- CPM reference: [arXiv:2505.23719](https://arxiv.org/abs/2505.23719)
- Benchmarks: [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) · [FEV-Bench](https://huggingface.co/spaces/autogluon/fev-bench) · [TIME leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard)
- Chronos-2: [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) · [technical report](https://arxiv.org/abs/2510.15821)
- 本笔记源码分析基于官方 `google-research/timesfm` commit `45e0a3bc7fc4acef17b7ba7910488be2159bae5f`。
- 本次核对时 Hugging Face `google/timesfm-3.0-pytorch` revision 为 `c71907076f28b1241d1fccc37efd183d0912cd13`。

> 状态说明：截至 2026-09-02，TimesFM-3 是 Google Research 技术文章 + 模型/代码 release，还没有一篇完整的 TimesFM-3 research paper。模型卡的 citation 仍指向 2023/2024 的原始 TimesFM 论文。因此本文会把博客 claim、公开代码事实、benchmark 复核和我们的判断分开写。

## 一句话结论

TimesFM-3 是一次实质升级：它把 TimesFM 从严格的 univariate foundation model 改成了**原生 multivariate foundation model**，并把 multiple targets、past-only covariates、known-future / past-future covariates 统一进同一个 Transformer；模型每层先做时间轴 causal attention，再做变量轴 full attention，同时用 Contiguous Patch Masking 把长 horizon 从逐 patch autoregressive decode 改成 single-pass forecast。

但它的学术 novelty 不能概括成“第一个支持 future covariates 的 foundation model”。Moirai 1.x 早在 2024 年就有 known-future dynamic covariate 路线，Chronos-2 在 2025 年也原生支持 known-future covariates。TimesFM-3 更准确的贡献是：**把 cross-variate attention、future-covariate lookahead 和 single-pass masked decoding 集成到一个强 zero-shot TSFM，并在多个公开 benchmark 上做到当前第一梯队。**

## 文章摘要

Google 在 2026-08-31 发布 TimesFM-3。官方文章强调，TimesFM 直到 2.5 都还是严格 univariate：预测每个 target 时只看该序列自己的历史。现实业务则经常同时存在：

- 多个相互影响的 target series；
- 只在历史阶段可见的 covariates；
- 在预测期也提前已知的 covariates，例如天气预报、促销计划、节假日和计划价格。

TimesFM-3 因此把输入组织成一个 `time × variate` 二维 token grid，让模型既沿时间学习单条序列的动态，也能在同一个时间位置跨 series 交换信息。官方模型有约 330M 参数，20 层 Transformer，model dim 1280、16 heads，输入 patch 长度 32、输出 patch 长度 64，公开实现最大 context 为 15,360。Google 称预训练 corpus 超过 1 trillion time points。

它同时直接输出 0.1 到 0.9 的 9 个 quantiles，所以 point forecast 和 probabilistic forecast 都来自同一个模型输出。

## TimesFM-3 到底做了什么

### 1. 原生 multivariate：不是把多条单变量预测拼在一起

核心 tensor 可以理解为：

```text
[batch, variate, time_patch, embedding]
```

每个 Transformer layer 的公开实现都是：

```text
sequence / temporal attention
        ↓
variate attention
        ↓
FFN
```

其中：

- temporal attention 沿 `time_patch` 维做严格 causal attention；
- variate attention 把 tensor reshape 成 `(batch * time_patch, variate, dim)`，在同一个时间位置对所有 variates 做 non-causal full attention。

因此某个 target token 可以直接读取同一时间位置的其他 targets / covariates，而不是每条序列先独立 forecast 再做后处理。

这也是 TimesFM-3 和 TimesFM-2.5 本质上最重要的结构差别。2.5 的核心模型仍是 univariate，后来加入的 covariate 支持主要经过 XReg；3.0 则在模型预训练和 attention architecture 里直接建模 cross-series dependency。

### 2. 原生 past-only 与 past-future covariates

公开 API 把 covariate 明确拆成两种：

```text
past_only_covariates
past_future_covariates
```

前者只有 context 区间；后者需要覆盖 `context + horizon`。

对于 target 与 past-only covariates，每个 token 直接从当前 patch 构造；对于 past-future covariates，模型会构造带 future lookahead 的 token：把当前 patch 和未来 patch 信息一起放进输入表示。这使已知未来事件能在 single-pass decode 时直接影响 target horizon。

因此“zero-shot”在这里的含义是**不需要为每个任务重新训练**，并不是“不需要业务变量”。如果未来天气、节假日、价格或排产计划是已知的，TimesFM-3 正是希望你把它们作为 past-future covariates 喂进去。

### 3. CPM：把 horizon 一次性 mask 后 single-pass 预测

TimesFM 以前的长 horizon 更接近：

```text
context → predict patch
              ↓ append
         predict next patch
              ↓
              ...
```

TimesFM-3 使用 Contiguous Patch Masking（CPM）：

```text
context + [MASK][MASK][MASK]...[MASK]
                   ↓
           one Transformer forward
                   ↓
              full horizon
```

具体来说：

- target future 全部 masked；
- past-only covariates 的 future 也 masked；
- past-future covariates 的 future 保持可见；
- alternating temporal/variate attention 在一次 forward 中同时生成整个 horizon。

这样有三个直接好处：

1. 减少 autoregressive latency；
2. 避免逐 patch error accumulation；
3. future-known covariates 可以在整个 horizon 上直接参与推理，而不是只能在每次 autoregressive step 局部注入。

### 4. RevIN、detrending 与 stitching 仍然很重要

TimesFM-3 并不是“只有 attention”。公开 `TimesFM3Torch` 还明确包含：

- patch-wise running statistics / RevIN；
- CPM 位置的 iterative RevIN refinement；
- optional linear detrending；
- overlapping patch prediction stitching；
- quantile head 与 quantile sorting；
- optional symmetric averaging / non-negativity clamp（benchmark evaluator 默认打开）。

这些属于不那么 headline、但会实质影响最终 benchmark 的 engineering / inference recipe。

## 代码实现分析

### 关键源码

3.0 的核心实现位于：

```text
src/timesfm3/
├── model.py
├── transformer.py
├── evaluator.py
├── timesfm3_forecaster.py
├── cpm_revin_refine.py
├── util.py
└── configs.py
```

另外官方把三套 benchmark runner 放在：

```text
timesfm3-usage/benchmarks/
├── fev_bench/
├── gift_eval/
└── time_bench/
```

这点不错：release 不只给模型权重，也把官方 benchmark inference wrapper 和逐任务结果文件放出来了。

### `model.py`：输入如何真正进入模型

`TimesFM3Torch.decode()` 接收：

```text
target:                  (b, num_targets, context)
past_only_covariates:    (b, num_po, context)
past_future_covariates:  (b, num_pf, context + horizon)
```

代码会先把三类 series 在 variate 维拼起来，再分别构造 context / horizon mask：

- target horizon = unknown；
- past-only horizon = unknown；
- past-future horizon = known；
- 最后 reshape 成 `(b, v, n_patches, patch_len)`。

这直接验证了博客所谓“native covariates”不是 API 层把变量传进去后再另做一个 regressor，而是进入模型主干的同一个 variate grid。

### `transformer.py`：真正的 2D mixing

`MixingTransformer` 的实现非常清楚：

```text
input: (b, v, n, d)

1. sequence attention
   reshape → (b*v, n, d)
   causal=True

2. variate attention
   permute/reshape → (b*n, v, d)
   causal=False

3. FFN
```

两条 attention 都有独立 RMSNorm / residual path。默认 `use_variate_attention=True`，而且这个 variate attention 出现在每一层 `MixingTransformer` 中。

所以这更接近 axial / factorized 2D attention：先沿时间轴混，再沿变量轴混，而不是对全部 `v × n` token 做一次 full attention。

### `evaluator.py`：一个容易被博客忽略的 32-variate 限制

官方 benchmark evaluator 定义：

```text
_MAX_VARIATES_PER_FORWARD = 32
```

如果 `targets + past-only covariates + past-future covariates > 32`，它会：

1. 先随机但固定 seed=42 地 subsample future covariates，最多保留 31 个；
2. 再 subsample past-only covariates；
3. 把剩余 target variates 分 chunk，多次 forward；
4. 最后把 target 输出拼回来。

这意味着“支持高维 multivariate”要加限定：**公开 evaluator 并不是任意多变量一次全局 cross-attention；32 个 variate 之外会通过 covariate subsampling + target chunking 近似处理。** 对超高维传感器、电网节点或股票 universe，这会直接改变模型看到的 cross-series context。

### context / patch 参数

公开实现中的几个关键常数：

- max context length: `15360`；
- input patch: `32`；
- output patch: `64`；
- 20 Transformer layers；
- model dim `1280`；
- 16 heads；
- 9 quantiles：`0.1 ... 0.9`。

Hugging Face safetensors metadata 给出的参数量为 `330,710,976`，和博客的“330M”一致。

## 和 TimesFM-2.5 的区别

| | TimesFM-2.5 | TimesFM-3 |
| --- | --- | --- |
| 参数量 | 200M | ~330.7M |
| 核心预测范式 | univariate | native univariate + multivariate |
| covariates | XReg 路径 | native past-only + past-future |
| cross-series attention | 无 | full variate attention |
| decode | patch-wise / iterative | CPM single-pass horizon |
| context | up to ~16k | implementation max 15,360 |
| 3.0 权重商用 | — | 不允许 |

因此 3.0 的升级价值不主要来自“从 200M 变成 330M”，而是训练任务和 inference architecture 都改了。

## Benchmark：哪些数字能直接比较

Google 的博客在 GIFT-Eval、FEV-Bench 和 TIME 上主要展示 **average rank across tasks**，并声称 TimesFM-3 在 point 与 probabilistic forecasting 上都是所有 pretrained foundation models 的第一。average rank 的优点是看跨任务稳定性，但它不能直接告诉我们 absolute error 到底改善多少。

### GIFT-Eval：同一套 97 configurations 的直接结果

GIFT-Eval 官方结果目录已经同时包含 TimesFM-3、Chronos-2、Moirai2、Toto 2.0、TimesFM-2.5、TabPFN-TS 和 TempoPFN。我们对官方 `all_results.csv` 按 97 个配置计算 geometric mean，得到：

| Model | 类型 | MASE ↓ | MWQL / CRPS proxy ↓ |
| --- | --- | ---: | ---: |
| TimesFM-3 | pretrained TSFM | **0.9321** | **0.1149** |
| Chronos-2 | pretrained TSFM | 0.9754 | 0.1224 |
| Toto-2.0-313M | pretrained TSFM | 0.9825 | 0.1214 |
| TimesFM-2.5 | pretrained TSFM | 0.9855 | 0.1236 |
| Moirai2 | pretrained TSFM | 1.0178 | 0.1302 |
| TabPFN-TS | ICL / PFN | 1.0776 | 0.1372 |
| TempoPFN | ICL / PFN | 1.1008 | 0.1343 |
| Moirai large 1.x | pretrained TSFM | 1.2229 | 0.1510 |

按这组同口径结果，TimesFM-3 相对 Chronos-2：

- geometric-mean MASE 约低 4.4%；
- MWQL 约低 6.1%。

相对 Moirai2 则分别约低 8.4% 和 11.8%。这属于有意义的领先，但不是断层式领先。

### ICL / PFN 路线

我们特别检查了 TabPFN-TS 和 TempoPFN，因为它们更接近“拿当前任务数据做 in-context adaptation”的 PFN / ICL 路线。

GIFT-Eval 上 TimesFM-3 相比 TabPFN-TS：

- MASE 约低 13.5%；
- MWQL 约低 16.3%。

相比 TempoPFN：

- MASE 约低 15.3%；
- MWQL 约低 14.5%。

所以至少在当前公开 forecasting benchmark 上，ICL/PFN 路线没有压过强 pretrained TSFM。

但这不是对 TabICLv2 的直接比较。当前 GIFT-Eval / FEV-Bench 都没有 TabICLv2 submission；TabICLv2 本身也不是原生 time-series forecaster。如果要比较，需要先把时序转换成 `lags + covariates → target` 的 tabular supervised problem，已经改变了输入范式和 benchmark protocol。

### FEV-Bench：Google 有 3.0 逐任务结果，但官方 leaderboard 尚未同步

截至本次核对，AutoGluon FEV-Bench 公开 leaderboard 仍显示：

```text
MASE win rate
Chronos-2       88.07%
TimesFM-2.5     75.07%
TabPFN-TS       58.68%

WQL win rate
Chronos-2       89.14%
TimesFM-2.5     78.29%
TabPFN-TS       68.11%
```

Google 的 TimesFM repo 已经放出 3.0 的 100-task FEV 结果文件并宣称 #1，但 FEV-Bench 官方 leaderboard 当时还没有 TimesFM-3 这一行。因此长期记录时应该把它写成：**Google release claim + 可下载逐任务结果已存在，但独立 leaderboard 同步仍待完成。**

### TIME

Google 也公开了完整 TIME benchmark runner，并说明使用 full native multivariate mode，与 Chronos-2 evaluation setup 对齐。博客声称在 98 tasks 上 point / probabilistic average rank 均第一。

但当前这篇技术发布没有像完整论文那样给出丰富的 ablation 表，所以暂时很难回答：

- variate attention 单独贡献多少；
- CPM 单独贡献多少；
- 330M / 更大预训练数据贡献多少；
- future-covariate lookahead 贡献多少。

这正是目前证据最大的缺口。

## “第一个支持 past-future covariates 的 foundation model”吗？

不是。

更合理的时间线至少包括：

```text
Moirai 1.x       2024   已支持 known-future dynamic covariates
Chronos-2        2025   原生 known-future real/categorical covariates
TimesFM-3        2026   原生 past-future covariates + variate attention + CPM
```

因此 TimesFM-3 的 novelty 不能写成“首次让 TSFM 使用未来已知变量”。

它更值得强调的是三件事的组合：

1. 多 target / covariates 共享同一个 variate-attention backbone；
2. future covariate token 带 lookahead；
3. CPM single-pass decode 让未来已知信号在整个 horizon 一次性参与预测。

## 和 Chronos-2 的关系

Chronos-2 是这次最值得直接比较的模型，因为它同样支持：

- univariate；
- multivariate；
- past-only covariates；
- known-future real / categorical covariates；
- zero-shot inference。

Chronos-2 是 120M encoder-only model，使用 group attention 做 cross-series / covariate ICL；TimesFM-3 是约 330M 的 alternating temporal + variate attention，并用 CPM single-pass 生成 horizon。

当前 GIFT-Eval 的绝对指标是 TimesFM-3 略优，但差距是个位数百分比，不应该解读成架构已经压倒性胜出。训练 corpus、synthetic data、模型规模和 inference recipe 都可能是 confounder。

### License 差异非常现实

这是实际落地时比几个百分点 benchmark 更重要的一点：

| Model | 代码 | 权重 | 商用 / production |
| --- | --- | --- | --- |
| Chronos-2 | Apache-2.0 | Apache-2.0 | ✅ |
| TimesFM-3 | Apache-2.0 | `timesfm-non-commercial-license-v1.0` | ❌ 当前默认权重不允许 |

Chronos-2 的 Hugging Face 模型卡和官方 `chronos-forecasting` repository 都明确是 Apache-2.0，可以用于商业和生产部署；TimesFM-3 源码仍是 Apache-2.0，但 3.0 pretrained weights 换成了 separate non-commercial license。

所以如果目标是业务落地，Chronos-2 当前的 license 明显更友好。

## 关键分析

### 1. 真正重要的是 native multivariate，不是“多传几个数组”

如果只是 API 支持 `X`，但模型内部仍按 univariate forecast 再做回归校正，那和 TimesFM-3 的性质不同。

TimesFM-3 的源码能确认：targets、past-only covariates 和 past-future covariates 最终都进入同一个 `(b, v, n, d)` Transformer grid，并通过 variate attention 交互。这个结构变化是实质的。

### 2. CPM 对 future covariates 特别契合

future covariate 最自然的问题是：预测期的已知信号怎么进入未来 target？

逐步 autoregressive decode 当然也能每一步塞，但 CPM 直接把整段 horizon placeholder 和 known-future channels 一起送进模型，让所有未来位置一次性在相同 context 中计算。这种推理设计与 past-future covariates 是互相加强的，不只是单独的 latency optimization。

### 3. benchmark 的提升不能全部归因于 multivariate

Google 特意报告了 TimesFM-3 univariate mode；博客称即使关掉 covariates / cross-series，3.0 也能匹配或超过主要竞争模型，再开 multivariate mode 后继续提升。

这意味着整体进步至少有两部分：

- 更强的 generalist backbone / pretraining / inference recipe；
- multivariate / covariate 信息带来的额外收益。

但目前没有完整 paper ablation，无法把两部分精确拆开。

### 4. 32-variate evaluator cap 是真实部署时必须记住的限制

在典型零售场景里十几个 target/covariates 可能够用；但在电力、工业 sensor、金融 cross-section 等高维场景里，32-variate cap 会导致：

- covariates 被 subsample；
- targets 被分块；
- 不同 chunk 的 target 之间不能完整互相 attention。

所以“native multivariate”不应该直接等价成“任意高维 joint model”。后续如果拿它做电力多节点预测，这一条要优先验证。

## Insights

### Insight 1：future-known covariates 已经从外挂功能变成 TSFM 的核心接口

TimesFM-3、Chronos-2、Moirai 的演进说明，真实 forecasting foundation model 正从：

```text
y_history → y_future
```

走向：

```text
targets_history
related_series_history
past_only_covariates
known_future_covariates
        ↓
joint forecast
```

这比单纯把 context length 从 2k 做到 16k 更贴近生产 forecasting 的真实需求。

### Insight 2：TSFM 与 tabular ICL 的边界开始值得认真比较

TabPFN-TS / TempoPFN 说明另一条路线是把 forecasting task 转成 ICL / PFN。当前 GIFT-Eval 上 TimesFM-3 更强，但 TabICLv2 这种更强 tabular ICL model 还没有在同口径 benchmark 上出现。

值得做的实验不是简单问“谁的 leaderboard 高”，而是构造严格相同的信息集：

```text
lags + past covariates + known-future covariates
```

分别交给：

- native TSFM；
- TabICLv2 / tabular ICL；
- strong GBDT / AutoGluon baseline。

这样才能判断 foundation forecasting architecture 本身到底带来多少超越强 tabular supervised formulation 的价值。

### Insight 3：average rank 适合看稳健性，不等于 absolute gain 很大

TimesFM-3 在三大 benchmark 都宣称 rank #1 是强信号，但 GIFT-Eval 对 Chronos-2 的绝对差距只有约 4–6%。因此“第一”可以成立，同时“不是断层领先”也成立。

以后看 TSFM release，应该同时保留：

- rank / win rate；
- normalized / absolute metric；
- task family breakdown；
- covariate / multivariate ablation；
- inference latency / memory；
- training overlap / contamination 信息。

## 我们的观点

### 这次升级比 TimesFM 2 → 2.5 更有意义

TimesFM-2.5 主要给人的感觉是更小参数、更长 context、quantile head 和 XReg 等工程升级；TimesFM-3 则真正改变了输入语义和主干模型：从独立单变量预测走到 joint multivariate forecasting。

因此 3.0 的“版本号”是有技术含量的，不只是规模/benchmark refresh。

### 但算法组件本身不是全新范式

如果拆成单项：

- axial / factorized attention 已经很常见；
- RevIN 不是新东西；
- known-future covariates 也早已有 TSFM 支持；
- masked horizon prediction / CPM 有独立前序工作。

TimesFM-3 的强点更像**组合、训练规模和 empirical result**：把这些组件做成一个统一、易用、强 zero-shot model，并在三个主流 benchmark 上给出很强结果。

所以当前更合理的判断是：

> 工程与 empirical contribution 很扎实，架构组合也合理，但在完整 paper / ablation 出来前，不宜把它包装成一个新的 forecasting paradigm。

### 对实际使用者：Chronos-2 仍然非常有竞争力

TimesFM-3 当前 GIFT-Eval 更强，但 Chronos-2：

- 只有 120M；
- 同样支持 multivariate + known-future covariates；
- Apache-2.0 权重可商用；
- 官方明确支持 production / SageMaker 路线。

因此“benchmark 最强”与“当前最值得生产采用”不是同一个问题。

## 值得继续追的问题

1. TimesFM-3 正式 paper 会不会补充 `variate attention / CPM / lookahead / data scale` 的独立 ablation？
2. 在只包含真正 covariate-informed tasks 的子集上，TimesFM-3 相比 Chronos-2 / Moirai2 的绝对增益是多少？
3. past-future covariates 的增益来自“未来值已知”本身，还是 TimesFM-3 的 lookahead token construction 真的优于普通 channel encoding？
4. 32-variate cap 放宽后，accuracy / memory / latency 怎么 scaling？
5. high-dimensional multivariate 场景下，target chunking 是否会破坏关键 cross-target dependencies？
6. TimesFM-3 的 single-pass CPM 在超长 horizon 上，相比 autoregressive decoding 的 error accumulation 改善有多大？
7. 如果用 TabICLv2 构造同信息集的 lag/covariate tabular benchmark，和 TimesFM-3 / Chronos-2 的差距会怎样？
8. Google 是否会提供可商用的 3.0 权重或把后续版本重新切回 permissive license？
