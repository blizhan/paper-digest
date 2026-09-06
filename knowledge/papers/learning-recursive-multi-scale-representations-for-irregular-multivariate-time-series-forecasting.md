---
type: Research Paper Review
title: "Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting"
description: "ReIMTS 的递归多尺度 IMTS forecasting 分析：按真实时间区间切分而非 resampling，重点讨论 IARF、跨尺度不等长表示融合、PyOmniTS 实现，以及与 Swin Transformer 相反的 top-down temporal pyramid。"
resource: https://arxiv.org/abs/2602.21498
tags: [irregular-time-series, multivariate-forecasting, multi-scale, representation-learning, reimts]
status: stable
sources:
  - id: paper
    resource: https://arxiv.org/abs/2602.21498
    title: "Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting"
  - id: openreview
    resource: https://openreview.net/forum?id=JEIDxiTWzB
    title: "ICLR 2026 OpenReview"
  - id: code
    resource: https://github.com/Ladbaby/PyOmniTS
    title: Ladbaby/PyOmniTS
---

# Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting

## Links

- Paper: [arXiv:2602.21498](https://arxiv.org/abs/2602.21498)
- PDF: [arXiv PDF](https://arxiv.org/pdf/2602.21498)
- ICLR 2026 / OpenReview: [JEIDxiTWzB](https://openreview.net/forum?id=JEIDxiTWzB)
- Official code: [Ladbaby/PyOmniTS](https://github.com/Ladbaby/PyOmniTS)
- 本笔记代码分析基于官方仓库 commit `47d8a2ff774ee12ccae451b3f11c5e778dca3902`。

## 一句话结论

ReIMTS 不是新的 IMTS backbone，而是一个递归多尺度 wrapper：**不通过 resampling / downsampling 构造粗尺度序列，而是按真实时间区间递归切分原始 irregular observations，再把粗尺度的 global representation 逐层传给细尺度。**

真正值得保留的设计点不是“用了多尺度”，而是把 `sampling pattern preservation` 当成硬约束。它通过时间区间切分、padding / mask、representation shape matching 和 gated cross-scale residual，解决了 irregular sequence 在不同尺度下 observation 数不等的问题。

## 文章摘要

Irregular multivariate time series（IMTS）的困难不只是 timestamp 不均匀，还包括不同变量在不同时间被采样的密度本身可能包含信息。论文认为，常见的 multi-scale 方法如果先做 interpolation、resampling 或 downsampling，会改变原始 timestamp / sampling pattern，尤其在医疗场景中可能抹掉“某个指标被更频繁观测”这种 signal。

ReIMTS 的方法是：

1. 按真实时间 period，而不是 observation 数，递归把完整时间范围切成更短的子区间；
2. 每个尺度使用同一种 IMTS backbone 架构提取 representation；
3. 把上一级 coarse/global representation 对齐到下一层的 local representation；
4. 用 IARF（Irregularity-Aware Representation Fusion）做 global → local 融合；
5. 最细尺度输出 forecast。

论文在 MIMIC-III、MIMIC-IV、PhysioNet'12、Human Activity、USHCN 五个数据集上测试，并把 ReIMTS 套到 PrimeNet、mTAN、TimeCHEAT、GRU-D、Raindrop、GraFITi 六个 backbone 上。作者报告平均 error reduction 为 `27.1%`，其中 PrimeNet + ReIMTS 的平均 MSE reduction 为 `62.3%`；GraFITi + ReIMTS 是整体最强组合。

## 核心设计

### 1. 不是 downsample，而是按真实时间区间递归切分

传统 multi-scale 更常见的是：

```text
fine-resolution sequence
        ↓ merge / downsample / resample
coarser sequence
        ↓
more global representation
```

ReIMTS 反过来从完整时间范围开始：

```text
完整 24h
├── 前 12h
│   ├── 前 6h
│   └── 后 6h
└── 后 12h
    ├── ...
```

这里发生的是 **split，不是 aggregation**。原始 observation 的 timestamp 和采样密度都不需要被改写。

因此不同 scale 的含义是：

- 大时间范围：偏 global dependency；
- 小时间范围：偏 local dependency；
- 每个子区间内部仍然保留原始 irregular observations。

这也是论文最核心的 inductive bias：**尺度由真实时间定义，而不是由“每段放多少个 observation”定义。**

### 2. 每层是同一种 backbone 架构，但不是共享同一套参数

从官方 PyOmniTS 当前实现看，ReIMTS 会递归创建多个 `Model(current_level=...)`，而每个 level 又单独实例化一次 backbone。

因此以 GraFITi 为例，更准确的结构是：

```text
GraFITi(level 0)
      ↓
GraFITi(level 1)
      ↓
GraFITi(level 2)
```

这些 level 使用同一种 backbone 类型，但属于不同模块实例，并不是一套 encoder 权重在不同尺度反复复用。

### 3. IARF：global → local 的 gated residual

论文把跨尺度融合抽象为 IARF：上一级 representation 提供 global information，当前层 representation 提供 local information，然后通过一个 learned score 做加权残差融合。

概念上可以写成：

```text
coarse/global representation H
              ↓
        learned score α
              ↓
local E + α · H
              ↓
        fused local G
```

对于包含 padding 的 time / observation representation，还需要结合 mask，避免把 padding 当成真实 observation。

需要注意论文描述和不同 backbone 的实际代码实现不应机械等同。当前 PyOmniTS 的 GraFITi 路径中实际使用：

```python
score = torch.sigmoid(self.score_layer(x_repr_var))
channel_embedding = channel_embedding + score * x_repr_var
```

所以对 GraFITi 来说，更具体地看就是一个 **sigmoid-gated residual connection**。

## 不同尺度序列不等长，代码如何融合

这是这篇最值得拆代码的地方。

### 1. 真实时间长度和 observation 数是两个概念

官方实现明确把两类长度分开：

```text
time_len_list
```

表示真实世界的时间长度，例如 48h / 24h；而：

```text
seq_len_max_irr
pred_len_max_irr
patch_len_max_irr
```

描述 irregular dataset 中沿 observation/time tensor 维度需要容纳的最大 observation 数。

例如同样一个 24h patch：

```text
patient A: 31 observations
patient B: 73 observations
patient C: 12 observations
```

这里 `24h` 是时间尺度，而 31 / 73 / 12 才是该区间真实 observation 数。

### 2. collate 阶段先按 timestamp 归属切区间

`collate_fn_fractal` 先把 lookback 和 forecast window 合并，再根据 timestamp threshold 判断每条 observation 属于哪个真实时间 patch。

因此它不是：

```text
每 20 个 observation 切一块
```

而是：

```text
0-12h 的 observation → patch 1
12-24h 的 observation → patch 2
...
```

某个 patch 里 observation 多就长，少就短，没有 observation 时也会明确生成空 patch / mask。

### 3. batch 内不等长靠 padding + mask

为了让 tensor 可以 batch 化，每个真实时间 patch 会 padding 到这一尺度允许的最大 observation 长度。

因此：

```text
[31 observations + padding]
[73 observations]
[12 observations + padding]
```

模型依靠 mask 区分真实值与 padding。

这不是 interpolation：padding 没有制造新的真实 timestamp 或 value，只是张量存储层面的 shape 对齐。

### 4. 尺度间 shape matching 的关键技巧：把子时间段塞进 batch dimension

`models/ReIMTS.py` 的 `patchify()` 使用 `einops.rearrange`：

```text
B (N_PATCH PATCH_LEN) V
        ↓
(B N_PATCH) PATCH_LEN V
```

例如：

```text
level 0: [32, 120, V]
       ↓ split into 2 children
level 1: [64, 60, V]
       ↓
level 2: [128, 30, V]
```

下一层 backbone 根本不需要理解 tree；它只看到：

- batch 变大；
- 每条 sequence 变短。

这是整个实现最工程化、也最简洁的部分。

### 5. 不同 representation 类型采用不同 shape matching

ReIMTS 允许 backbone 输出三类 representation。

#### Temporal representation

```text
[B, L, D]
   ↓ split / patchify
[B*N, L/N, D]
```

#### Observation representation

同样沿 time/observation 维做 split。

#### Variable representation

```text
[B, V, D]
   ↓ repeat_interleave
[B*N, V, D]
```

因为 variable representation 没有时间维，无法切成前 12h / 后 12h，所以直接给每个 child time period 复制一份 coarse variable representation。

因此论文里的 `SplitOrDuplicate` 在当前代码中就是非常直接的：

```text
time / observation representation → split
variable representation           → duplicate
```

## PyOmniTS 真实代码路径

当前官方实现可以压缩成：

```text
collate_fn_fractal
      │
      │ 按真实 timestamp 分时间区间
      │ 每段按 observation 数 padding + mask
      ↓
ReIMTS(level=0)
      │
      ├─ backbone_0
      │      ↓
      │   global repr
      │
      ├─ patchify(x)
      ├─ split / duplicate(global repr)
      ↓
ReIMTS(level=1)
      │
      ├─ current repr + gated coarse repr
      │
      ├─ backbone_1
      │
      ├─ patchify
      ↓
ReIMTS(level=2)
      │
      └─ backbone_2
             ↓
          prediction
```

核心文件：

- `models/ReIMTS.py`
  - recursive level construction；
  - `time_len_list` / `time_len_max_irr_list`；
  - `patchify()` / `unpatchify()`；
  - coarse representation 的 split / duplicate；
- `data/dependencies/tsdm/PyOmniTS/tsdmDataset.py`
  - `collate_fn_fractal`；
  - 根据 timestamp 切真实时间 patch；
  - 计算 irregular patch 最大 observation 长度；
- `layers/ReIMTS/models/*.py`
  - 不同 backbone 对 coarse representation 的接入；
- `layers/ReIMTS/layers/GraFITi/GraFITi_layers.py`
  - GraFITi 路径中的 variable representation fusion / sigmoid gate。

## 关键实验结果

### Backbone 兼容性

ReIMTS 被套在六个结构差异很大的 IMTS backbone 上：

- PrimeNet
- mTAN
- TimeCHEAT
- GRU-D
- Raindrop
- GraFITi

论文报告跨模型 / 数据集平均 error reduction 为 `27.1%`，PrimeNet 的平均 MSE reduction 为 `62.3%`，GraFITi + ReIMTS 的总体结果最好。

这部分实验主要支持的是：**ReIMTS 更像一个通用 multi-scale wrapper，而不是只对单一 backbone 有效的专用 architecture。**

### 时间切分 vs observation-count 切分

以 GraFITi backbone 的 ablation 为例，MSE（论文表中为 ×10^-1）：

| Method | MIMIC-III | MIMIC-IV | PhysioNet'12 | Human Activity | USHCN |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReIMTS | 4.07 | 1.79 | 2.83 | 0.42 | 1.66 |
| replace sample | 4.99 | 1.92 | 2.83 | 0.45 | 1.69 |
| 按 observation 数切分 | 5.02 | 2.36 | 3.20 | 0.61 | 2.31 |
| IARF 改成直接 addition | 4.20 | 1.84 | 2.79 | 0.47 | 1.89 |
| 去掉 IARF | 4.77 | 2.07 | 3.06 | 0.54 | 1.69 |

其中最有说服力的是“按 observation 数量切分”明显变差，这直接支持论文最核心的 design choice：**尺度应由真实时间 period 定义，而不是 observation count。**

但需要保留一个证据 nuance：在 PhysioNet'12 上，“IARF 改成直接 addition”得到 `2.79`，反而略好于 ReIMTS 的 `2.83`。因此不能把结果写成“IARF 在每个数据集上都严格占优”；更准确的是，它在总体实验中有稳定价值，但不是每个 dataset / 每个配置都单调改善。

### scale 数量不是越多越好

论文的 scale sensitivity 也很重要。USHCN 上：

```text
2 scales: 1.66
3 scales: 1.80
4 scales: 2.01
```

多数数据集最优只需要两层，MIMIC-III / PhysioNet'12 才更偏好多尺度。

因此 ReIMTS 不是“无脑增加 hierarchy 就能涨点”。论文中的时间尺度仍然依赖 domain knowledge，例如医疗数据使用日周期附近的尺度，USHCN 使用年 / 半年尺度。

### 计算成本

ReIMTS 不是免费提升。以 MIMIC-III 的 GraFITi 路径为例，论文报告单 iteration 从约 `33ms` 增加到 `66ms`，GPU memory 从约 `0.390GB` 增加到 `0.598GB`。

但相比 Warpformer / Hi-Patch 等专门 multi-scale architecture，ReIMTS 的显存仍明显更低；这与它“保持 backbone 简单，只递归包装”的工程定位一致。

## 与 Swin Transformer：多尺度构造方向基本相反

这是我们讨论中一个很有用的类比。

### Swin Transformer

```text
fine/local patches
      ↓ patch merging
coarser representation
      ↓
larger receptive field
      ↓
global
```

可以概括为：

```text
local → regional → global
```

### ReIMTS

```text
完整长时间范围
      ↓ split
较短时间范围
      ↓ split
更短时间范围
      ↓
local
```

同时 coarse/global representation 会不断传给 child level，因此信息流更接近：

```text
global → regional → local
```

所以从 hierarchy construction 的方向看，两者可以理解为相反：

| | Swin Transformer | ReIMTS |
| --- | --- | --- |
| 起点 | fine / local | long-range / global |
| 尺度方向 | fine → coarse | global → local |
| 主要操作 | patch merging | time-period splitting |
| token / observation | 聚合、减少 | 原 observation 保留 |
| receptive range | 逐层扩大 | 逐层缩小时间窗口 |
| 跨尺度信息 | bottom-up | top-down |

但不能把 ReIMTS 的“大尺度”理解成“低分辨率序列”。例如 48h level 仍保留 48h 内所有真实 irregular observations，它只是覆盖范围更大，并没有把多个 observation merge 成一个 token。

因此更准确的类比是：

> **Swin 是 bottom-up spatial pyramid；ReIMTS 是 top-down temporal pyramid。**

两者最终目的其实相同：同时利用 local + global dependencies，只是 hierarchy 的构造方向和信息压缩方式不同。

## 关键分析

### 1. 真正的新意不是 multi-scale 本身，而是 sampling-pattern-preserving multi-scale

如果只说“递归多尺度”，这个设计并不复杂。ReIMTS 的价值在于把一个容易被忽略的约束放到了中心：

> 不要为了构造尺度而改变原始 irregular sampling pattern。

尤其医疗数据中，采样密度可能是 informative missingness / clinical behavior 的一部分。先做规则化处理虽然方便模型，但可能同时删掉了有用信号。

### 2. 它更像 framework / wrapper，而不是新的基础 backbone

从代码看，真正的方法层集中在：

- 时间尺度定义；
- fractal-style recursive split；
- shape matching；
- global → local fusion；
- backbone adapter。

单个融合模块并不复杂，核心实现也没有引入一个庞大的新 neural architecture。

因此我们更倾向把其创新归为 **representation / pipeline design**，而不是基础模型架构创新。

### 3. “不插值”不等于“不需要张量对齐”

一个容易误解的点是：ReIMTS 虽然避免 interpolation / resampling，但它仍然必须为了 batch 计算做 padding、zero-mask 和 tensor alignment。

二者区别是：

```text
interpolation / resampling
→ 改变或生成数据语义上的观测点

padding / mask
→ 只解决存储和 batch shape，不声明这些位置是真实观测
```

所以“保留原始 timestamp”并不意味着模型内部永远处理 ragged list；实际实现仍然会张量化。

### 4. scale 仍然是人工设计的

当前方法没有 adaptive scale discovery。不同数据集使用什么 period、需要几层，都带有明显 domain prior。

这也是后续最值得继续追的方向之一：能否让模型自动发现有效 temporal periods，而不是人工传入 `patch_len_list`。

## Insights

### Insight 1：irregular sampling pattern 本身可以是一种 feature

IMTS 处理中不能默认“缺失 / 不规则只是数据质量问题”。什么时候观测、多久观测一次、哪些变量更密集地被观测，本身可能是预测信号。

因此任何 regularization preprocessing 都应该问：

> 它是不是在让输入更整齐的同时，也把 informative sampling behavior 抹掉了？

### Insight 2：multi-scale 不一定需要 downsampling

ReIMTS 提供了另一种构造尺度的方法：

```text
scale = observation aggregation
```

并不是唯一选择；也可以是：

```text
scale = different temporal support / context range
```

这在 irregular / event stream / sparse sensor 数据上尤其值得保留。

### Insight 3：hierarchy 可以 top-down，而不一定 bottom-up

视觉模型常见的是 local token 逐层合并成 global token；ReIMTS 则说明，对于 forecasting，可以从 global context 出发，把 global representation 逐层注入更局部的时间段。

这给其他时序结构一个很自然的设计空间：

```text
global context encoder
        ↓ condition
mid-range encoder
        ↓ condition
local high-resolution decoder
```

### Insight 4：mixed-frequency forecasting 与 ReIMTS 有联系，但不能画等号

我们之前讨论过“低频气象 → 高频风电，不先把低频信号插值到高频”。ReIMTS 可以作为一个重要的设计论据：**不要仅仅为了输入 shape 方便就提前 regularize 不同频率或 irregular signals。**

但原论文解决的是 general IMTS forecasting，不等于已经解决：

```text
low-frequency exogenous covariate
           ↓
high-frequency target forecasting
```

mixed-frequency exogenous-to-target 的 causal alignment、future covariate availability、不同变量频率之间的 decoder 设计，仍然需要额外建模。


## 风电 mixed-frequency 扩展：Future Weather Encoder + Query Decoder

补充行业 PDF《拒绝插值，从低频气象直接预测高频风电功率》把 ReIMTS 思路扩展到了一个更直接的 mixed-frequency forecasting 场景：**小时级未来 NWP + 15 分钟历史功率 → 15 分钟未来功率**。

需要明确区分：下面的输入输出接口与 query 定义来自 PDF；`CrossAttention` 等 decoder 内部实现是我们根据接口给出的合理伪代码，PDF 并未披露实际网络结构。

### 任务设定

```text
未来 NWP：1h，一天 24 点
历史风电功率：15min，一天 96 点
预测目标：未来 24h 的 15min 功率，共 96 点
```

传统方案会先做：

```text
24 点 NWP
   ↓ PCHIP / linear / spline / forward fill
96 点 NWP
   ↓ predictor
96 点 power
```

该方案则保留 NWP 与功率各自原始时间戳，不预先生成 96 点天气：

```text
24 点原始 NWP + 96 点历史功率
              ↓
      分别编码 / 多尺度融合
              ↓
       96 target-time queries
              ↓
         Query Decoder
              ↓
       96 点未来功率
```

### Future Weather Encoder

PDF 明确说明 15 分钟历史功率与小时级未来气象分别编码，因此概念接口可以写成：

```python
H_power = PowerEncoder(
    values=power_hist.values,       # 概念 shape: [B, 96, C_power]
    timestamps=power_hist.time,
    mask=power_hist.mask,
)

H_weather = WeatherEncoder(
    values=weather_future.values,   # 概念 shape: [B, 24, C_weather]
    timestamps=weather_future.time,
    mask=weather_future.mask,
)
```

这里 `[B, 96, C]` / `[B, 24, C]` 是我们为了说明接口写出的合理 tensor shape，并非 PDF 披露的具体实现。PDF 也没有说明 Weather Encoder 究竟使用 Transformer、GRU、GraFITi 还是其他 backbone。

真正重要的是它**没有**：

```python
weather_96 = interpolate(weather_24, freq="15min")
```

而是直接把原始 24 个 NWP 时点编码成 `H_weather`。

### 96 个 target-time query 决定输出频率

PDF 给出的查询时点定义为：

\[
q_k = \frac{k-1}{4}, \qquad k=1,\dots,96
\]

对应：

```text
q1  = 0.00h
q2  = 0.25h
q3  = 0.50h
...
q96 = 23.75h
```

伪代码：

```python
Q = []
for k in range(96):
    target_time = k / 4.0
    Q.append(TimeEmbedding(target_time))

Q = stack(Q)  # conceptually [B, 96, D]
```

因此**输出频率由 query grid 决定，而不是由未来气象的输入频率决定**。从模型接口上说，24 → 96 并不要求一定是固定 4 倍升频；理论上可以查询任意 target timestamps。

### Query Decoder

PDF 明确给出的接口是：

\[
\hat y_{1:96} = D(Q_{1:96}, H_{\text{power}}, H_{\text{weather}})
\]

也就是：

```text
                  H_power
                     │
Q_1...Q_96 ──────────┼──→ Decoder ──→ ŷ_1...ŷ_96
                     │
                  H_weather
```

PDF 没有进一步说明 `D` 是 cross-attention、Transformer decoder、MLP 还是其他结构。一个自然但**仅属于我们的实现推断**的版本是：

```python
predictions = []

for q in Q:
    weather_context = CrossAttention(
        query=q,
        key=H_weather,
        value=H_weather,
    )

    power_context = CrossAttention(
        query=q,
        key=H_power,
        value=H_power,
    )

    z = concat(q, weather_context, power_context)
    y_hat = PredictionHead(z)
    predictions.append(y_hat)

forecast = stack(predictions)  # conceptually [B, 96, 1]
```

这个实现的关键不是 attention 本身，而是：**每个目标时点主动从低频 weather representation 和历史 power representation 中读取所需上下文，而不是先在数据空间造出一个 15 分钟天气值。**

### 更准确地理解“拒绝插值”

严格来说，它不是取消了所有 frequency mapping，而是把显式、固定的 data-space interpolation 改成了由预测目标学习的 latent-space temporal querying。

传统方式：

\[
\tilde x(t) = g(x_h, x_{h+1}, t), \qquad
\hat y(t) = f(\tilde x(t), p)
\]

其中 `g` 是线性、PCHIP、样条等人工规则，与最终功率 loss 无关。

query decoder 更接近：

\[
z(t) = g_\theta(t, H_{\text{weather}})
\]

\[
\hat y(t) = f_\theta(z(t), H_{\text{power}}, t)
\]

所以更技术准确的说法是：

> **不在数据空间做手工升频，而是在 latent space 做 target-aware temporal querying。**

### ReIMTS 与 Query Decoder 的职责不同

这两部分不应该混为一谈：

```text
ReIMTS
→ 解决 context resolution
→ 24h / 6h / 1h 等 global → local multi-scale context

Query Decoder
→ 解决 output resolution
→ 根据任意 target timestamp 生成预测
```

因此，一个普通 encoder + query decoder 理论上也可以做到 `24 点天气 → 96 点功率`。要证明风电实验中的收益确实来自 ReIMTS，而不仅是“raw mixed-frequency input + query decoding”，更公平的 ablation 应该至少包括：

```text
A. interpolation + predictor
B. raw 24-point NWP + query decoder
C. ReIMTS + query decoder
```

这也是当前行业 PDF 最明显的证据缺口之一。

### 可以抽象成更通用的多源异频预测接口

风电案例最终可以抽象为：

```python
predict(
    observations=[
        Source(power, freq="15min"),
        Source(weather, freq="1h"),
        # satellite / station / price / other sources can keep own timestamps
    ],
    target_times=[
        "00:00", "00:15", "00:30", ..., "23:45"
    ],
)
```

也就是：

```text
heterogeneous time-series sources
           ↓
per-source / shared timestamp-aware encoders
           ↓
optional multi-scale context fusion
           ↓
arbitrary target-time queries
           ↓
target-frequency output
```

这个抽象比“所有源先 resample 到 15min”更适合多 NWP、卫星、站点观测、SCADA、价格等原生频率不同的数据源。

## 我们的观点

这篇的研究含金量高于“把已有 multi-scale 搬到 irregular TS”这么简单的描述，但算法复杂度并不高。

它最好的地方是三个设计都很克制：

```text
真实时间 hierarchy → split
不等长 observation → padding + mask
跨尺度表示 → gated residual
```

并且用把 child patch 放进 batch dimension 的方式，把递归 hierarchy 落成一个普通 backbone 很容易接入的工程结构。

如果已经有 GraFITi、mTAN、GRU-D 等 IMTS backbone，ReIMTS 的价值更多在于**提供一种相对低侵入的 multi-scale upgrade path**，而不是要求完全换模型。

我们目前对其主要弱点的判断是：

- scale period 依赖 domain knowledge；
- scale 数量需要调参，且多并不一定好；
- IARF 本身非常简单，部分数据集上直接 addition 并不差；
- 参数不共享意味着 scale 增多时模型实例数也增加；
- 它没有直接解决 mixed-frequency exogenous → high-frequency target forecasting。

总体上，这是一篇**设计思想比模块复杂度更有价值**的论文。

## GitHub / Code Analysis

官方 PyOmniTS 当前代码对论文的核心设计基本给出了直接实现，而且比论文图更容易看清楚。

### 递归模型构造

`models/ReIMTS.py` 中：

```python
self.time_len_list = [configs.seq_len + configs.pred_len] + configs.patch_len_list
```

如果当前不是最细层，会直接递归创建：

```python
self.next_model = Model(
    configs=configs,
    current_level=current_level + 1
)
```

同时每层都会动态 import 并实例化一次 backbone，因此同类型 backbone 不共享权重。

### patchify

核心 reshape：

```python
rearrange(
    x,
    "B (N_PATCH PATCH_LEN) ENC_IN -> (B N_PATCH) PATCH_LEN ENC_IN",
    N_PATCH=...
)
```

把 tree child dimension 折叠进 batch，是整个递归实现保持简单的关键。

### representation 传递

当前代码中：

```python
if "pred_repr_time" in backbone_output:
    input_dict["x_repr_time"] = self.patchify_repr_time(...)

if "pred_repr_var" in backbone_output:
    input_dict["x_repr_var"] = torch.repeat_interleave(...)

if "pred_repr_obs" in backbone_output:
    input_dict["x_repr_obs"] = self.patchify_repr_time(...)
```

这直接对应论文的 `SplitOrDuplicate`。

### GraFITi 的融合

GraFITi 返回 `channel_embedding` 作为 `pred_repr_var`；下一层 encoder 在存在 `x_repr_var` 时做：

```python
score = torch.sigmoid(self.score_layer(x_repr_var))
channel_embedding = channel_embedding + score * x_repr_var
```

因此从当前代码事实看，GraFITi 路径中的 cross-scale fusion 是一个简单 gated residual。

### 数据侧 fractal collate

`collate_fn_fractal` 明确区分：

- `patch_len`：真实时间 unit；
- `patch_len_max_irr`：该真实时间 patch 内实际 observation 数的最大值。

这正是为什么 irregular sample 可以在不插值的情况下进入固定 shape tensor。

## 值得继续追的问题

1. **Adaptive scale discovery**：能否从数据中学习 24h / 12h / 6h 这类 period，而不是人工指定？
2. **Parameter sharing**：不同 level backbone 完全独立是否必要？共享部分参数能否降低成本同时保留收益？
3. **IARF 必要性**：既然部分数据集上简单 addition 已接近甚至略优，真正需要 learned gate 的条件是什么？
4. **Sampling-pattern ablation**：能否更直接地量化收益到底来自 multi-scale context，还是来自 preserving informative sampling density？
5. **Mixed-frequency extension**：如何把这种“不提前 regularize”的思想扩展到低频 exogenous covariate → 高频 target？
6. **更大规模 / 非医疗数据**：USHCN 已显示小样本和多尺度之间存在冲突，方法在更大工业 sensor / event stream 上是否仍稳定？
