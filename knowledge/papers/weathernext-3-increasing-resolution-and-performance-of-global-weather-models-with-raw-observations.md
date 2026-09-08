---
type: Research Paper Review
title: "WeatherNext 3: Increasing resolution and performance of global weather models with raw observations"
description: "WeatherNext 3 的多模态全球概率天气预报分析，重点讨论低延迟卫星小时级初始化、0.1°/1h 输出、连续站点 head、观测监督、相对 WN2 的真实增益与评测限制。"
resource: https://arxiv.org/abs/2609.03582
tags: [weather-forecasting, probabilistic-forecasting, multimodal-learning, observations, satellite]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2609.03582
    title: "WeatherNext 3: Increasing resolution and performance of global weather models with raw observations"
  - id: project
    resource: https://deepmind.google/science/weathernext/
    title: WeatherNext 3
  - id: code
    resource: https://github.com/google-deepmind/weathernext
    title: google-deepmind/weathernext
---

# WeatherNext 3: Increasing resolution and performance of global weather models with raw observations

## Links

- Paper: [arXiv:2609.03582](https://arxiv.org/abs/2609.03582)
- HTML: [arXiv HTML](https://arxiv.org/html/2609.03582v1)
- Project / forecast access: [Google DeepMind WeatherNext 3](https://deepmind.google/science/weathernext/)
- Official code repository: [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)
- 本笔记代码检查基于官方仓库 commit `f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3`。该版本尚未包含可识别的 WN3 model/config 实现，公开代码仍以 WN2、WeatherNext Cyclones、GraphCast 和 GenCast 为主。

## 一句话结论

WeatherNext 3 真正重要的变化不是简单把 WeatherNext 2 做大，而是把 **analysis、低延迟 geostationary satellite、卫星降水产品、稀疏地面站和 cyclone targets 放进同一个 probabilistic global forecasting system**：主模型仍以 6 小时 outer autoregressive step 推进，但原生预测小时级变量，并利用最新卫星观测做到每小时重新初始化。

这使它同时跨过了三条过去相对分离的边界：global medium-range forecast、observation-aware initialization、以及 station / precipitation post-processing。不过“raw observations”需要谨慎理解：WN3 **并不是从原始观测完全替代传统 data assimilation**，它仍然大量依赖 ERA5/HRES analysis / short forecast 作为 atmospheric state，卫星观测更像是低延迟增量信息；站点与降水 observation 则主要作为额外 target/head 进入统一训练。

## 文章摘要

论文针对当前 global AI weather model 的两个限制：一是主流 AI forecast 的时空分辨率仍低于最强 NWP system，二是绝大多数模型只从 analysis 初始化并只预测 analysis，因此继承 analysis 的偏差和 6 小时更新周期。

WN3 延续 WN2 的 Functional Generative Networks（FGN）probabilistic formulation，输出 15 天、64-member ensemble，但扩展成一个多模态、多分辨率系统：

- 0.25° atmospheric fields + 0.1° single-level analysis；
- 0.1°、每小时 11-channel geostationary satellite mosaic；
- 0.1°、每小时 PARDIG / IMERG precipitation targets；
- sparse station observations，连续查询 2m temperature / dewpoint；
- tropical cyclone existence / track / intensity / extent targets。

结果上，WN3 相对 WN2 在 2024 全年 analysis evaluation 中大部分变量继续提升；在独立站点与降水观测上增益更大；低延迟卫星带来的 hourly refresh 对短时降水大约换来 2–3 小时有效 lead-time gain。生产模型在 2026-07-01 到 2026-08-11 的 6 周实时评测中也整体领先 AIFS ENS v2，但这部分样本较短，并存在 training recency、ground truth 和 regridding 差异，需要比 headline 更谨慎解释。

## 这篇到底做了什么

### 1. 从单一 analysis modality 扩展成多模态 forecast state

WN2 基本只建模 0.25° analysis variables。WN3 给不同 native resolution / cadence 的数据分别使用 encoder / decoder，再映射到共享的 icosahedral processor mesh：

```text
0.25° atmospheric analysis ─┐
0.1° surface analysis      ─┤
0.1° hourly satellite      ─┤→ shared icosahedral processor → modality-specific decoders
sparse station metadata    ─┤
cyclone targets            ─┘
```

这样不同 modality 不需要先全部 regrid 成一个共同分辨率，processor mesh 负责跨模态交互和时序 propagation。

### 2. 每小时重新初始化，关键来自低延迟卫星

传统 global analysis 通常只有 6 小时 cycle，并且真正可用时已经有明显 operational latency。WN3 的 geostationary satellite mosaic 每小时更新，论文给出的实际延迟约 1 小时，比最新 analysis 至少新约 5 小时。

模型仍然输入 analysis，但会叠加最近的 satellite frames。论文举的例子是：14 UTC 做预测时，最新卫星帧来自 13 UTC，可输入 08–13 UTC 的卫星；最新 HRES analysis cycle 仍是 06 UTC，因此取其 valid at 08 UTC 的短时 forecast state。

这点很关键：

> WN3 的 hourly initialization 是“较旧的 analysis state + 更新鲜的 observation stream”，不是纯 observation-to-forecast 的完整 learned data assimilation replacement。

卫星通道还被 autoregressively predicted，使其信息能够跨多个 6h outer rollout steps 继续传播。

### 3. 小时级输出是原生模型变量，不再靠单独 temporal upsampler

WN2 的 6h forecast 到 hourly product 需要额外的 `6h → 1h` upsampler。WN3 仍以 6h outer step autoregress，但把一个 6h window 内的若干 hourly substeps 当作独立 variables 一次预测，因此：

- single-level variables 原生 1h cadence；
- 部分 pressure-level variables 也提供 hourly output；
- 其他 atmospheric variables 仍以 6h cadence 为主。

也就是说，WN3 的“1h”不是把整个 dynamical core 改成逐小时 autoregressive rollout，而是 **6h outer dynamics + window 内多 subtime targets**。

### 4. Station head：把稀疏 observation 变成 continuous query

站点 head 是这篇很有价值的一部分。模型把 0.1° latent 插值到任意 query location，再拼上：

- exact elevation；
- land / sea metadata；
- 相对 6h outer step 的 continuous time offset；

然后预测 2m temperature 和 dewpoint。

因此从模型接口上可以在任意经纬度、任意时间查询。论文生产与评测实际生成的是 0.05° / hourly grid；直接在精确站点坐标查询只带来很小额外增益。

训练站点来自 METAR、Mesonet、ICOADS；METAR / Mesonet 随机但时间一致地 hold out 5% stations 用于 spatial generalization evaluation。由于真实站点覆盖很不均匀，作者还每小时随机加入 2,000 个由 ERA5/HRES 插值得到的 pseudo-stations，降低 Andes、Himalayas 和高纬海洋等 OOD 区域偏差。

这同时也是一个重要 caveat：station head 虽然直接用 observation supervision，但为保证全球空间行为，仍重新注入了 analysis-derived pseudo labels。

### 5. Precipitation：把目标从 analysis precipitation 换到更接近观测的产品

WN2 只预测 analysis precipitation。WN3 新增：

- **PARDIG**：Google 自己的实验 precipitation estimate，由另一个 AI model 从 GPM Core Observatory 的 sparse space-borne radar signal 学习得到；
- **IMERG Final**：NASA GPM 的常用全球卫星降水产品。

两者都是 0.1° / 1h targets。论文内部分析认为 PARDIG 比 IMERG 更接近独立 ground radar / rain gauge，因此主推 PARDIG head。

这里也不要把 PARDIG 等同于完全 raw radar observation：它本身就是另一个 learned precipitation product。

### 6. Backbone 和 training 也明显扩容

相对 WN2：

- latent size：`768 → 1024`；
- mesh transformer depth：`24 → 32`；
- training resolution curriculum：`1° → 0.25° → 0.1°`；
- ERA5 history：从 WN2 的 1979 起扩展到 1959 起；
- rollout finetuning：逐步从 2AR 到 8AR；
- station head 在 0.1° backbone 上 frozen finetuning；
- 为高分辨率引入 processor mesh / gather-scatter spatial sharding。

因此论文里相对 WN2 的 skill gain **不能全部归因于卫星输入**。模型容量、训练数据长度、training curriculum、native resolution 和目标数据都同时变了，缺少足够完整的 ablation 去把每一项拆开。

### 7. Epistemic dropout 替代一部分 deep ensemble 成本

WN2 用 4 个独立 model seeds 表达 epistemic uncertainty；WN3 只训练 2 seeds，同时加入 epistemic dropout。

训练时，同一步里的 ensemble members 共享一个 dropout mask；推理时，每个 member / timestep 独立采样 mask。作者的动机是避免训练样本把 dropout variance 当作免费 ensemble spread，同时在 inference 时扩展 epistemic spread。

这个设计帮助两 seed 的更大模型维持 ensemble spread，但 cyclone evaluation 里仍出现比 WN2 更明显的 under-spread，说明它并没有完全解决 joint uncertainty calibration。

### 8. FGN：不是 diffusion，而是 `state + noise → forecast member`

WN3 延续 WN2 的 Functional Generative Networks（FGN）。对固定天气状态 `x`，同一个共享 backbone 接收不同随机变量 `ξ`，直接产生不同 ensemble member：

```text
weather state + ξ₁ → forecast member 1
weather state + ξ₂ → forecast member 2
weather state + ξ₃ → forecast member 3
...
```

它与 GenCast 的 diffusion sampling 路线不同：FGN 不需要在每个 member 上做多步 denoising，而是一次 stochastic forward 就得到一个样本，因此更适合 operational large ensemble。

公开 WN2 代码把随机性实现为一个名为 `noise` 的 global conditioning feature。`gaussian_noise_generator` 默认生成 white Gaussian noise；当前 WN2 配置中是 `32` 个 noise channels。这个 noise 不直接粗暴 concat 到天气场，而是进入每层的 normalization conditioning。

### 9. Fair CRPS：直接优化 ensemble distribution，而不是每个 member 各自 MSE

公开 WN2 FGN 代码中的核心 loss 是 CRPS：

```text
CRPS = MeanAbsError - 0.5 * MeanAbsDiff
```

即：

```text
accuracy term:
    member 到 truth 的平均绝对误差

diversity term:
    member 与 member 的平均绝对距离
```

更标准地写：

```text
CRPS(F, y)
  = E|X - y| - 1/2 E|X - X'|
```

其中 `X, X'` 是从同一预测分布独立采样的两个 forecast samples。代码默认 `unbiased=True`，pairwise term 用 finite-ensemble 的 fair / unbiased estimator，而不是普通 empirical CRPS。

这和普通 ensemble training 的差别非常大：如果所有 members 坍缩成同一个预测，第一项可能仍然不错，但第二项不能提供 ensemble spread；CRPS 会直接把“预测准确”和“分布有合理宽度”同时放进 objective。

WN3 论文进一步说明，训练时并不需要真的生成 64 members。实际只采 **两条 stochastic trajectories** 来估计 fair CRPS；64 members 是推理时为了更细致地表示 forecast distribution。

两个 sample 已经足够，是因为 CRPS 本身只需要估计两个期望：

```text
E|X-y|
E|X-X'|
```

两条独立 trajectory 是计算第二项的最小数量。每一步虽然只看一对随机样本，但训练过程中不断重采 `ξ`，本质上是在用 stochastic Monte Carlo gradient 估计完整 distribution-level objective，而不是声称两个样本可以在单次 forward 中完整描述 64-member 分布。

### 10. Normalization modulation：随机 latent 如何真正控制大模型

FGN 最关键的实现技巧不是“有一个 32-d Gaussian latent”，而是这个 latent **如何持续影响整个深层网络**。

普通 LayerNorm 后通常是固定可学习参数：

```text
h' = γ ⊙ LN(h) + β
```

FGN 把 `γ, β` 改成由随机 condition `ξ` 生成：

```text
γ = γ(ξ)
β = β(ξ)

h' = γ(ξ) ⊙ LN(h) + β(ξ)
```

公开代码里的 `LinearNormConditioning` 会把 conditioning 投影成 `2 * hidden_dim`，一半作为 scale、一半作为 offset：

```python
conditional_scale_offset = Linear(2 * feature_size)(norm_conditioning)
scale_minus_one, offset = split(conditional_scale_offset)
scale = scale_minus_one + 1.0
return inputs * scale + offset
```

这个 modulation 会在 Transformer block 的 attention / FFN 路径中反复施加，因此不同 `ξ` 不是只在 input 端造成一次很弱的扰动，而是控制整条 hidden computation trajectory：

```text
ξ
↓
small conditioning network
↓
{γ₁, β₁}, {γ₂, β₂}, ... {γL, βL}
↓
shared large backbone
↓
different plausible forecast
```

另一个很漂亮的稳定性设计是 conditioning linear layer 用极小初始化（公开实现中 `stddev=1e-8`），并通过 `scale = scale_minus_one + 1` 让训练初始时：

```text
γ(ξ) ≈ 1
β(ξ) ≈ 0
```

也就是一开始 modulation 几乎什么都不做，模型先接近 deterministic forecast，再让 CRPS 逐渐学出“哪些 latent variation 对应哪些合理未来”。这比一开始就让随机 condition 大幅扰乱整个网络稳定得多。

### 11. 它是不是在学习一个 forecast manifold？只能谨慎地说“implicit family”

固定天气状态 `x` 后，FGN 定义了一个映射：

```text
g_x: ξ → fθ(x, ξ)
```

输入 `ξ` 只有几十维，而输出是百万维级别的未来天气场。如果这个神经映射局部连续，那么其 image 自然形成一个低维参数化的 forecast surface，因此可以把它直观理解成 **implicit forecast manifold / learned ensemble family**。

但 FGN 并没有显式 manifold constraint，也不保证：

- latent mapping 是 injective；
- 拓扑正确、没有 self-intersection；
- latent distance 对应天气语义距离；
- 物理守恒；
- 所有 plausible futures 都被覆盖。

所以更准确的说法不是“FGN 保证学到了真实 weather manifold”，而是：

> FGN 用低维连续随机变量，通过 layer-wise modulation 参数化一个共享 backbone 上的 forecast function family；CRPS 再迫使这个 family 同时贴近 truth 并保持足够 spread。

### 12. 和 TabM / Mapping Networks 的联系

这次讨论里一个很有价值的抽象，是把 FGN 放到“member-specific freedom”这条轴上看：

```text
参数共享高                                             参数共享低

TabM                  FGN                      Mapping Networks
│                     │                        │
固定 K 个 member      连续 ξ                   连续 z
少量固定 modulation   生成每层 γ/β             生成大量/全部 weights
│                     │                        │
finite ensemble       implicit infinite family weight/model manifold
```

- [TabM](tabm-advancing-tabular-deep-learning-with-parameter-efficient-ensembling.md)：通常是固定 `K` 个成员，每个成员保存少量 parameter-efficient modulation；
- FGN：`ξ` 连续，理论上可生成无限 members，但绝大多数 backbone weights 仍共享，只生成 layer-wise normalization modulation；
- [Mapping Networks](mapping-networks.md)：latent 可以决定更大范围、甚至完整的模型权重，更接近显式 weight-space manifold。

因此 FGN 可以看成一个很自然的中间点：**continuous TabM + lightweight hypernetwork / mapping network**。它没有生成一整套新模型，却让连续 latent 在网络深度上持续改变函数行为。

另一个值得迁移的思路是：如果以后在 TabM / packed ensemble 上做概率预测，不一定需要每步把全部 members 放进显存；可以像 FGN 一样随机抽 member pair，并用 proper scoring rule / CRPS 直接训练 ensemble family，而不是只做逐 member CE/MSE 后再平均。

## 关键实验结果

### 2024 全年、训练截止 2023 年底

这是论文里相对最干净的一组主结果。

| Evaluation | 主要结果 |
| --- | --- |
| 0.25° upper-level analysis | 相对 WN2 大多数变量提升；medium-range 平均约 `5%`，作者换算成同 skill 下约多 `6h` lead time |
| 0.1° surface analysis | WN3 在所有 surface variables 上优于 WN2；2m temperature 提升尤其明显 |
| unseen stations: 2m temperature | 短 lead time CRPS 相对 WN2 最多约 `30%` 降低，相对 ENS 最多约 `40%` |
| unseen stations: 10m wind | early lead CRPS 约 `5%` 降低 |
| precipitation / PARDIG head | early lead CRPS 相对 global probabilistic baselines 最多约：IMERG `60%`、MRMS `30%`、rain gauges `10%` 降低 |
| hourly refresh | latency-adjusted precipitation skill 相比只用 6-hourly initialization 获得约 `2–3h` lead-time gain |
| tropical cyclones | track / intensity MAE 小幅但稳定改善；1–3 day extent 改善更明显，但 intensity / extent ensemble 更 under-spread |
| SSRD / FDIR | 相对 ECMWF ENS 多数 lead time 更低 CRPS；从论文 scorecard 近似读取，Day 1–5 SSRD 大致约 `10–15%` 改善，FDIR 可到约 `10–20%`，但作者没有提供独立逐值表 |
| TCC / HCC / MCC / LCC | 多数 Day 1–7 优于 ENS，scorecard 大致约 `5–15%` CRPS 改善；最初 `6–12h` 部分云变量反而会退化 |

Precipitation 的最大提升发生在 rain/no-rain 附近的低降水阈值；论文明确没有充分评估 extreme precipitation，高阈值统计太稀疏，因此把 extreme precipitation evaluation 留给后续工作。

### SSRD / 云：对新能源更重要，但证据要分清楚

WN3 原生输出 `surface_solar_radiation_downwards`（SSRD）、direct solar radiation，以及 total / high / medium / low cloud cover。论文 Figure 2 的 2024 scorecard 中，这些变量相对 ECMWF ENS 大多数 lead time 都更好，而且提升幅度通常明显高于普通 upper-air 的代际 `~5%`。

从论文色阶图近似读取：

- SSRD：Day 1–5 大致约 `10–15%` CRPS 改善；
- FDIR：前几天大致约 `10–20%`；
- TCC / HCC / MCC / LCC：多数 Day 1–7 大致约 `5–15%`。

这些不是作者提供的精确 numeric table，因此只能作为图表近似值，不能写成严格 benchmark 数字。

更值得注意的是 lead-time structure：cloud 在最初几个小时并不稳定占优，论文 Appendix 的前 48h 曲线中，部分 TCC/HCC/MCC/LCC 在 `6h`、有时 `12h` 反而比 ENS 差，之后才稳定反超。对光伏业务意味着目前更有把握的优势区间是 **Day 1–5**，而不是直接假设 `0–6h / 0–12h` 一定更好。

从机制上看，cloud / SSRD 又恰好是最可能从 hourly satellite conditioning 中受益的变量之一：

```text
最新 geostationary satellite
    ↓
云位置 / 云顶温度 / 水汽 / 云系演变
    ↓
shared latent
    ↓
TCC / HCC / MCC / LCC
    ↓
SSRD / FDIR
```

但当前论文的主 SSRD / cloud ground truth 仍主要来自 HRES-style analysis / forecast targets，而不是独立 pyranometer、BSRN、电站 GHI 或独立 satellite cloud product。因此目前能比较有把握地说：

> WN3 比 ECMWF ENS 更会预测 ECMWF-style SSRD / cloud target。

不能直接推出真实电站 GHI / PV power 也会同步提高 `10–15%`。如果用于新能源业务，最有价值的下一步是拿公开 WN3 hourly `SSRD + cloud` 产品直接对自己的辐射站 / 电站功率做 independent verification。

另一个技术信号是论文专门为 cloud 增加了 global-pooled CRPS 辅助项，并在 inference 时把 cloud clip 到 `[0,1]`、solar radiation clip 到 `>=0`。这再次说明 point-wise marginal CRPS 很强，但 individual member 的 global / joint structure 仍可能存在系统性偏差。

### 2026 实时 6 周评测

生产 WN3 使用训练到 2026-06-30 的 checkpoint，在 2026-07-01 到 2026-08-11 与 operational WN2、AIFS ENS v2 做 quasi-real-time comparison。

- upper-level variables：前一周 forecast range 内相对 AIFS ENS 平均约 `10%` CRPS 改善；
- station temperature / humidity：WN3 station head 显著优于 AIFS ENS；
- 0.1° analysis wind speed：站点评测约 `5%` CRPS 改善；
- precipitation：PARDIG 对 MRMS 明显最好；对 rain gauges，PARDIG 与 IMERG heads 大致相当，并明显优于 AIFS ENS / WN2 / ENS。

但这组结果的证据强度低于 2024 full-year：只有 6 周，而且 WN3 与 AIFS ENS v2 都接触过 ECMWF 50r1 data，WN2 没有；AIFS 和 WN3 的 ground-truth / interpolation path 也不同，作者明确承认会影响分数。

## 相比 WeatherNext 2 的实质变化

| 维度 | WeatherNext 2 | WeatherNext 3 |
| --- | --- | --- |
| 主输入 | analysis | analysis + hourly geostationary satellite |
| 主要 grid | 0.25° | 0.25° atmosphere + 0.1° surface / observation modalities |
| hourly output | separate 6h→1h upsampler | native subtime targets inside 6h outer step |
| forecast refresh | 主要 6-hourly cycles | hourly initialization |
| precipitation target | analysis precipitation | analysis + PARDIG + IMERG |
| station target | 无专门 continuous station head | 2m temperature / dewpoint continuous query head |
| cyclone | 已有 WN Cyclones 路线 | 统一到更大的 WN3 multimodal model |
| latent / transformer | 768 / 24 layers | 1024 / 32 layers |
| epistemic ensemble | 4 independent seeds | 2 seeds + epistemic dropout |
| training history | ERA5 from 1979 | ERA5 from 1959 |

最值得保留的理解是：**WN3 的进步来自“统一模态 + 更高 native resolution + observation supervision + 低延迟 refresh + scaling”的组合，不是一项单独 architecture trick。**

## 关键分析

### “从 raw observations 预报”这个 headline 有营销压缩

论文确实第一次在这一代 operational global model 中大规模直接 ingest geostationary observations，并且直接监督 precipitation / station observations；这是实质变化。

但 atmospheric backbone 依然依赖 ERA5 / HRES，hourly initialization 也是在 analysis / short forecast state 上叠加更近的 satellite stream。因此更准确的说法是：

> WN3 是 analysis-anchored、observation-augmented global forecast model，而不是已经彻底替代传统 analysis / data assimilation 的 end-to-end raw-observation forecaster。

### 最大创新更像“forecast state API 被重新定义”

过去 global model 的接口大致是：

```text
analysis grid → future analysis grid
```

WN3 更接近：

```text
analysis + observations + metadata
    → future gridded atmosphere
    → future satellite / precipitation products
    → arbitrary-location station predictions
    → cyclone table-like outputs
```

这比单纯提高 WeatherBench score 更重要，因为它开始把 data assimilation、forecast 和 post-processing 之间的边界压进一个 shared latent system。

### 站点 head 的价值比“0.05°”这个数字更大

站点 head 本质上不是固定 0.05° super-resolution network，而是一个基于 latent interpolation + geography metadata 的 continuous decoder。0.05° 只是论文当前 production/evaluation materialization grid。

这意味着同一个 backbone 可以服务：

- 任意站点查询；
- 不同分辨率的下游产品；
- 理论上的 site-specific forecast API。

它更接近把传统 MOS / post-processing 内化成主模型的一部分。

### 论文最缺的是完整 ablation

WN3 相对 WN2 同时改变：model size、data history、native resolution、satellite input、training targets、training curriculum、uncertainty mechanism。论文虽然通过 common-grid evaluation 证明提升不只来自 resolution，也通过 hourly-vs-6-hourly initialization isolate 了一部分 refresh gain，但仍没有完整回答：

- satellite input 单独贡献多少；
- 1024 / 32-layer scaling 单独贡献多少；
- 1959–1978 额外 ERA5 history 贡献多少；
- observation targets 的 multi-task regularization 对 analysis score 有多大作用；
- PARDIG quality 与 WN3 model quality 各贡献多少 precipitation gain。

因此“某个设计导致 SOTA”这类因果性解读证据不足。

### Joint sample artifacts 是一个真实结构问题

WN3 训练优化的是 marginal CRPS，因此 individual ensemble members 的 covariance structure 可以出现作弊空间。论文主动展示了：

- mesh-related hexagonal artifacts，尤其在 precipitation / station outputs 明显；
- station head 在 6h outer-step boundary 存在 temporal jumps；
- individual station ensemble member 有 global warm/cold bias，并在每个 6h step 重采 noise 后变化；
- cyclone intensity / extent 比 WN2 更 under-spread。

median、quantile、threshold probability 等 marginal / ensemble statistics 受影响较小，但如果下游要使用单 member trajectory、spatiotemporal coherence、flood/hydrology coupling，这些 artifact 不能忽略。

## 评测限制与 confounders

1. **Analysis ground truth 不是独立 observation。** 主要 analysis score 用 HRES-fc0，而 HRES family 也是模型训练和初始化的重要来源。
2. **IMERG 不是独立 ground truth。** WN3 本身把 IMERG 当 target；甚至 PARDIG head 因 end-to-end co-training 也可能间接受到影响。MRMS / rain gauges 的独立评测因此更有说服力。
3. **ENS baseline 数据源存在 TIGGE / MARS 差异。** 作者明确说这会轻微惩罚 ENS，虽然估计不足以改变主结论。
4. **2026 AIFS comparison 只有 6 周。** 而且 training recency 和 regridding / ground-truth path 不完全一致。
5. **Station spatial coverage 很偏。** 训练集中大量站点来自机场和欧美 Mesonet，pseudo-stations 缓解 OOD bias，但也重新把 analysis bias 引入 station head。
6. **Extreme precipitation 尚未真正覆盖。** 当前 headline precipitation skill 主要支持一般到中等降水，不足以直接推出极端暴雨能力。
7. **Cyclone joint calibration 退化。** point skill 提升不等于 ensemble spread 同样提升。

## Insights

### 1. AI weather 下一阶段很可能不是“更强 analysis emulator”

WN3 给出的方向很清楚：global forecast model 的输入输出 schema 会越来越像 heterogeneous observation system，而不是固定 ERA5 tensor。

真正的竞争点会逐渐从：

```text
谁在同一套 analysis benchmark 上 RMSE 更低
```

转向：

```text
谁能利用更新鲜、更异构的 observations，
并直接输出最接近用户决策 grain 的 forecast products。
```

### 2. Latency 本身就是 forecast skill

hourly initialization 并没有神奇地创造 2–3 小时 dynamics skill；它主要把信息 freshness 优势变成实际 decision-time skill。

这提醒我们，部署型天气模型应该把：

- source latency；
- initialization cadence；
- model runtime；
- dissemination latency；

一起算进有效 lead time，而不是只比较 nominal forecast lead。

### 3. Continuous decoder 是比固定高分辨率 grid 更有扩展性的 product interface

station head 说明高分辨率 local forecast 不一定要把整个 global state 都显式 materialize 到极细网格。可以把共享 latent 保持在较合理的 resolution，再通过 query-aware decoder 按需生成 local product。

这对气象服务的存储、API 和下游产品设计都很有启发。

### 4. Multi-target observation training 可能比继续堆 analysis variables 更重要

PARDIG、IMERG、station、cyclone 都在迫使 shared latent 同时解释不同 observation systems。即使论文没有足够 ablation 证明 multi-task regularization 的具体贡献，这条路线值得追：**让 latent 对多个真实 measurement spaces 同时可解码，可能比单一 reanalysis imitation 更接近真实世界 forecast representation。**

## 我们的观点

这篇的研究价值很高，而且比“WN2 再涨几个点”重要得多。最值得跟的是 **observation-aware global forecasting + continuous product heads**，因为它开始改变 AI weather model 的系统边界。

但标题里的 “with raw observations” 容易让人误以为它已经完成 observation → global state → forecast 的 fully learned data assimilation。实际不是：analysis 仍是 backbone，satellite 提供 freshness，station / precipitation 提供更接近 ground truth 的 supervision。更准确地说，WN3 是从 “analysis emulator” 朝 “heterogeneous forecast model” 迈出了一大步。

如果后续出现真正从 radiance / station / radar / satellite 等 observations 直接建立 global latent state，并在 operational medium range 上达到同等级别 skill 的模型，那才是对传统 data assimilation 边界更彻底的替代。

## GitHub / Code Analysis

检查官方仓库 `google-deepmind/weathernext` commit `f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3`：

- `README.md` 仍主要描述 WeatherNext 2 / WeatherNext Cyclones；
- model implementation 目录包含 `weathernext2/architecture.py`、`weathernext2/fgn.py`、cyclone tracking、GraphCast / GenCast 等；
- 仓库中搜索 `WeatherNext 3` / `WN3` / `weathernext3` 没有命中；
- 因此当前不能把 WN2 公开代码直接当成 WN3 implementation 做逐源码对照。

论文描述的 shared icosahedral processor、modality-specific encoder/decoder、spatial sharding、continuous sparse decoder 等与现有 WeatherNext codebase 的基础组件有明显家族关系，但在 WN3 代码真正公开之前，只能把这些当论文事实，不能声称公开仓库已经提供对应实现。

### 公开 WN2 FGN 代码确认了哪些机制

虽然 WN3 implementation 尚未公开，但 WN2 的 FGN 核心实现已经能确认我们讨论的概率生成机制不是只来自论文文字：

- `weathernext/weathernext2/fgn.py`
  - `gaussian_noise_generator` 生成随机 condition；
  - `crps_loss` 明确实现 `CRPS = MeanAbsError - 0.5 * MeanAbsDiff`；
  - 默认 `unbiased=True`，pairwise spread term 使用 fair estimator；
  - 对 autoregressive residual targets 做 shift，确保 pairwise MAD 对应 forecast weather states，而不是不同 initial state 下的 residual 差异；
  - target 为 NaN 的位置会同步 mask prediction，避免模型通过在无监督位置无限放大 ensemble variance 来“hack”CRPS。
- `weathernext/utils/sparse_transformer.py`
  - Transformer block 在 attention / FFN 前反复调用 norm conditioning，因此随机 condition 会贯穿整个 processor depth。
- `weathernext/utils/dense.py`
  - `LinearNormConditioning` 由 condition 生成 channel-wise scale / offset；
  - conditioning weight 使用 `TruncatedNormal(stddev=1e-8)` 小初始化；
  - `scale = scale_minus_one + 1.0` 让初始 modulation 接近 identity。
- `weathernext/weathernext2/configs/WeatherNext2.json`
  - `noise` 被列为 `norm_conditioning_features`；
  - 当前公开配置的 `noise_channels = 32`。

这些代码事实可以支持“FGN 是连续 latent → layer-wise modulation → forecast family”的解释，但不能反推 WN3 的所有实现细节都与 WN2 完全一致。WN3 论文明确延续 FGN，同时又加入 epistemic dropout、更多 modality 和新的 composite loss，因此最终 stochastic space 更复杂。

### 当前开放性：还不能把自己的自动站 / 卫星直接接进 WN3

截至本次阅读时，官方开放的是 WN2 / Cyclones 等代码与 checkpoint，以及 WN3 的 forecast products；**WN3 自身的训练图、checkpoint、satellite encoder、station decoder 和 observation ingestion pipeline 尚未公开**。

因此目前不能直接做：

```text
自己的 AWS 自动站 ─┐
自己的卫星数据 ────┼→ 官方 WN3 model → 新 forecast
                   ┘
```

现实可行的两条路线是：

1. 把公开 WN3 forecast 当 global background，使用自己的自动站 / 电站数据做 calibration、MOS、downscaling 或 site-specific probabilistic post-processing；
2. 基于公开 WN2/FGN 代码复现一个 regional observation-conditioned model，再自己实现 satellite encoder / sparse station head。

对于东亚 / 中国区域，一个更实际的研究原型可以是：

```text
ERA5 / ECMWF background
        +
FY-4 / Himawari satellite
        +
AWS / radiation stations
        ↓
regional observation-conditioned FGN
        ↓
1h probabilistic cloud / SSRD / site forecasts
```

这条路线比复刻全球 WN3 更现实，也能直接验证我们真正关心的问题：低延迟卫星和本地观测是否能显著提升 `0–48h` 云 / 辐射 / 光伏概率预报。

### 计算成本

论文报告：

- 每个 WN3 seed 训练约 `6.7 days` wall-clock；
- 两个 seeds 合计约使用 `4.8 TPUv4 chip-years + 8.9 TPU7x chip-years`；
- 单个 ensemble member 的 15-day forecast 大约需要 `6.3 min` wall-clock，使用 `4 TPUv5p chips`。

这也说明 WN3 虽然推理相对传统 NWP 仍然便宜，但已经不是“小模型 AI weather”路线，训练和 operational ensemble serving 都是明显的大规模基础设施项目。

## 值得继续追的问题

1. WN3 code / checkpoint 是否会公开，以及哪些 modality heads 会开放本地 inference。
2. 做严格 ablation：satellite input、resolution scaling、model scaling、extra ERA5 history、observation multi-task heads 分别贡献多少。
3. 能否把 HRES/ERA5 analysis anchor 进一步移除，真正做 observation-native global initialization。
4. PARDIG 是否会公开数据/模型；它本身对 precipitation score 的贡献有多大。
5. extreme precipitation、convective storms、snow 等高影响天气在独立 observation 上的 skill。
6. 如何直接优化 spatiotemporal joint distribution，消除 mesh artifacts 和 6h boundary discontinuity，同时保留 marginal CRPS skill。
7. continuous decoder 能否扩展到风、降水、辐射、能源 site forecasts，形成真正 query-native weather API。
8. hourly forecast refresh 在真实 operational decision benchmark 上，相比单纯 nominal lead-time metric 能带来多大价值。
