---
type: Technical Article Review
title: "Introducing Timestral"
description: "Timestral 技术发布与源码分析：DHF 双流预测、角色约束的双轴 attention、合成数据训练及相对 TimesFM-3 的小幅领先；结合电价预测说明协变量可得性，并审视第三方 DuckDB 扩展的能力边界。"
resource: https://mistral.ai/news/timestral/
tags: [time-series-foundation-model, forecasting, multivariate, covariates, synthetic-data, duckdb]
status: stable
sources:
  - id: article
    resource: https://mistral.ai/news/timestral/
    title: Introducing Timestral
  - id: code
    resource: https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
    title: mistralai/timestral
  - id: model-card
    resource: https://huggingface.co/mistralai/Timestral-209M
    title: Timestral-209M model card
  - id: weights-license
    resource: https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE
    title: Timestral weights Modified MIT License
  - id: duckdb-extension
    resource: https://github.com/altertable-ai/duckdb-timestral/tree/6936c3957551ec6df255b19c47b6cf822b05efb8
    title: altertable-ai/duckdb-timestral — third-party extension
---

# Introducing Timestral

## Links 与资料状态

- 官方技术文章 / 项目入口：[Introducing Timestral](https://mistral.ai/news/timestral/)，发布于 2026-10-15。
- 官方代码：[mistralai/timestral](https://github.com/mistralai/timestral)。实际阅读对应 commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`。
- 官方权重与结果：[Timestral-209M](https://huggingface.co/mistralai/Timestral-209M)；另有 5M、28M、123M。
- 第三方集成：[altertable-ai/duckdb-timestral](https://github.com/altertable-ai/duckdb-timestral)。实际阅读对应 commit `6936c3957551ec6df255b19c47b6cf822b05efb8`。
- 关联笔记：[TimesFM-3](timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting.md)、[Chronos-2](chronos-2-from-univariate-to-universal-forecasting.md)。

截至本次核对（2026-10-26），官方 model card 的 technical report 仍标为 coming soon；本篇记录的是**技术博客、model card、公开推理源码与我们的讨论**，不是已经读过完整论文。没有已确认的 Timestral arXiv / PDF 链接，不拿旧论文或猜测编号代替。本文未运行模型、复现 benchmark 或安装扩展。

## 一句话结论

Timestral 把独立 forecast-query tokens（DHF）、按变量角色限制的信息流、合成数据训练和一套完整推理配方组合成强 zero-shot TSFM。值得关注的是：它如何让未来协变量参与预测，同时阻止真实 target 信息经双向协变量通道倒流，以及合成多变量 prior 到底贡献了多少泛化能力。

作者公布的结果相对 TimesFM-3 是跨 benchmark 一致但幅度较小的领先；现有资料不足以把收益单独归因于 DHF。官方 OSS 提供完整的协变量推理接口，但没有完整训练流水线；第三方 DuckDB 扩展目前仅暴露分组单变量预测，不能等同于完整 Timestral 能力。

## 文章摘要

官方面向的问题是：在新领域不重新训练模型，也能同时利用多个目标的历史关系、历史辅助观测和提前可得的未来信息，输出概率预测。其方法采用 patch Transformer，以时间轴和变量轴 attention 组合传递信息，用独立预测 token 查询历史，再生成分位数。

博客描述的训练先在纯合成数据上进行，context 为 8,192；之后在真实与合成数据混合上后训练，context 扩到 32,768。合成过程先生成单变量动态，再引入非线性依赖；目标是 21 个分位数的 pinball loss。具体生成器、混合比例与训练消融尚无法从公开推理仓库复核。作者在 GIFT-Eval、TIME、fev-bench 报告领先，并通过 Forge 提供领域适配。这些训练说明来自[官方博客](https://mistral.ai/news/timestral/)，不应写成已验证的开源训练代码事实。

## 方法拆解：这篇到底做了什么

### 1. 表示与双轴 attention

公开 209M 配置说明为 24 层、embedding dim 768，每个 patch 包含 32 个时间步；每步输入同时包含数值和 missing indicator。最多读取 32,768 个历史时间步，单个 decode block 最多输出 32 patches，即 1,024 步。

`PatchRepresentation` 把每条变量序列的 `[values, missing]` patch 映射成 token。模型内部采用 packed `(R, N, D)`，`R` 是不同样本各条变量展开后的行，`group_ids` 确定哪些行属于同一个样本。

- Time attention：沿一条变量的时间 patches 运行，带 RoPE。
- Variate attention：在相同 token slot 上跨变量运行，受 group 与 role mask 限制，不加变量位置编码。
- 层执行顺序由 `block_schedule` 指定，支持 time、variate 及两种组合；不能机械理解为每层都固定执行完全相同的两个子层。

来源：[model.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/model.py)、[layers.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/layers.py)、[model card](https://huggingface.co/mistralai/Timestral-209M)。

### 2. DHF：把历史表示与预测查询分开

博客对 Contiguous Patch Masking 路线的批评是：被遮掉的观测既成为预测目标，也从后续预测的 context 中消失，还与真实缺失共享相似的输入状态。DHF 的设计动机是把这些职责分开。

```text
Enc tokens：真实观测的 patch 表示，保留历史信息
Dec tokens：learned forecast queries，按 mask 读取允许的 Enc / Dec 信息
                          ↓
                    quantile head
```

源码中的 `[Enc ; Dec]` 是同一个 Transformer 内的两组 token，并非独立训练的两个 encoder/decoder 网络。`forecast_query` 是共享的可学习向量，位置由时间 position / RoPE 区分。

无协变量路径 `_predict_context_only` 中，Enc 采用 causal mask；每个 Dec 查询可以读有效历史 Enc 和自身及更早的 Dec 查询，Enc 不读 Dec。一个 block 内的 Dec 通过一次网络计算得到预测，不需要把每个已生成数值逐 patch 回喂。跨 block 才反馈预测中位数。

这能在推理源码中确认双流表示及访问规则，但不能据此复现完整训练时的 DHF sampling / loss，也不能证明 DHF 对 CPM 的净收益。来源：[model.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/model.py)、[masks.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/masks.py)。

### 3. 三类变量如何交互

三类变量的身份主要参与 mask 构造，而不是进入三个独立模型。

| 角色 | 观测 Enc token 的时间访问 | 参与预测的方式 |
| --- | --- | --- |
| Target | Causal：读取当前及更早的有效 patches | 使用历史观测，输出未来分位数 |
| Past-only covariate | Causal | 提供历史辅助信息，未来空缺，不作为最终预测输出 |
| Known-future covariate | 双向：读取窗口内已提供的有效值 | 可将预测期已知信息传到 target |

变量轴的访问权限如下，**行读取列**；还需满足同一个 `group_id`、相应 key validity 等条件。

| 读取方 ↓ / 被读取方 → | Target | Past-only | Known-future |
| --- | --- | --- | --- |
| Target | 允许 | 允许 | 允许 |
| Past-only | 禁止 | 允许 | 允许 |
| Known-future | 禁止 | 禁止 | 允许 |

这张表来自 `variate_firewall`：所有 covariate 禁止读 target；具有双向时间访问权限的 known-future row 还禁止读任何 causal row。因此 past-only 的隐藏表示可以融合 known-future 信息，但它仍然只是 conditioning row。

一个可成立的信息路径是：

```text
未来风速预报
  → known-future 的双向 time attention
历史位置的天气表示汇集未来信息
  → 相同位置的 variate attention
历史电价表示融合天气信息
  → forecast query 读取历史表示
未来电价分位数
```

这里的 time/variate attention 需经过多层接力；不是任意未来天气 token 都通过一次 target temporal attention 直接可达。`supports_known_covariates` 还检查层顺序是否具备传递这些信息的必要结构。

为什么 known-future 不能读其他 causal row？否则，较晚的真实电价或历史辅助观测可以先进入同位置天气表示，再沿天气的双向时间 attention 倒流，绕过 target 自己的 causal mask。Firewalls 旨在阻断这种模型内部的间接泄漏。**它不能保证上游数据的发布时间正确，也不代表 attention 学到的是因果效应。**

来源：[masks.py：time masks / variate_firewall](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/masks.py)、[config.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/config.py)。

### 4. 概率输出与推理配方

`QuantileHead` 输出 21 个分位数。`monotone_quantiles` 以自由中位数为中心，通过 softplus 正增量向上下累加，显式保证分位数不交叉；不交叉不等于覆盖率已校准。

Target / past-only 采用截至各 patch 末尾的因果统计量做标准化，并按配置使用 asinh；known-future 使用整条输入窗口的统计量。后者之所以允许非因果统计，是因为该行未来输入已在 cutoff 时可得。

默认 `forecast` 还包含：

- **Symmetric averaging**：分别预测原输入及整体取负的输入，把后者按 `-Q(1-p)` 反射回来逐分位数平均；covariates 也取负。
- **Non-negative clipping**：某条 target 的已观测历史全部非负时，将最终预测裁到非负。对可能出现负价的电力市场，这个隐含先验需要专门评估。
- **Advancing refine frame**：后续预测 patch 的反标准化坐标系结合前面预测中位数推进；另有极值保护及 quantile cap。
- **长 horizon rollout**：Target 反馈预测中位数，known-future 接真实提供的未来协变量，past-only 接缺失；超出 context cap 后丢弃最旧历史。

所以单 checkpoint 不等于只运行一次未经增强的网络。作者所称无 ensemble，不应被解释成完全没有 test-time averaging。跨 block 反馈中位数也不等于完整传播前面预测的不确定性。

多变量联合表示最终输出 `(B, V, H, 21)` 的边际分位数；当前接口没有直接提供保留变量/时间依赖的联合轨迹采样，不能把这些分位数当成完整联合概率分布。

来源：[forecaster.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)、[model.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/model.py)、[scaler.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/scaler.py)。

## 我们的讨论：以电价预测解释变量角色

假设今天 18:00 为 cutoff，预测明天 00:00–24:00 每 15 分钟电价。以下是任务建模示例，不是 Timestral 的电力实验结果；具体市场的发布时间另行核对。

| 数据 | 角色 | 为什么 |
| --- | --- | --- |
| cutoff 前已发布的历史电价 | Target 的历史部分 | 要预测电价，其历史值是目标上下文 |
| 明天真实电价 | Target label | 事后评估，不能作为输入 |
| 历史实际新能源出力、实际负荷 | Past-only | 历史可得，预测期实际值未知 |
| 历史实测风速、气温、辐照度 | Past-only | 提供历史环境，需考虑发布延迟 |
| cutoff 前发布的明日新能源出力预测、负荷预测 | Known-future | 曲线已可用，即使存在误差 |
| 今天 12:00 发布的明日气象预报 | Known-future | 未来真实天气未知，但这份预报的值已知 |
| 明天事后记录的实测天气 | 不能作为本次未来输入 | 会把未来真实信息带入回测 |
| 日历、节假日、已公布检修计划 | Known-future | cutoff 时已经能获得预测期取值 |

**Known-future 描述的是输入的可得性，不是预报一定准确。** 同一物理量也可换角色：若联合预测电价与新能源出力，实际出力就是另一个 target；未来出力预报仍可作为单独协变量。

### 历史气象预测：区分 issue time 与 valid time

```text
issue_time：今天 12:00（这份预报何时发布）
valid_time：明天 14:00（预报描述哪个时刻）
forecast_temperature：28℃
cutoff：今天 18:00
```

`issue_time <= cutoff` 说明这条预报当时可用；`valid_time` 位于预测区间，说明它可以提供未来条件。更早发布、描述历史区间的预报则属于历史侧输入。历史回测必须逐 cutoff 从 archive 选择当时已有的版本，不能用后来的修订版或实测值替换。

官方 `known_covariates` 需要覆盖 `(B, C_known, T + H)`；`past_covariates` 为 `(B, C_past, T)`。实际可将实测气温单独放 past-only，气温预报放 known-future，明确历史预报的 lead time / 版本选择规则。不要在没有定义的情况下，将历史实测与未来预报视为误差特性相同的一条变量。示例的角色划分不能替代时间对齐与数据可得性设计。

### Categorical 是另一种输入类型，不是第四种时间角色

类别特征也可以是 past-only 或 known-future：历史机组状态可能仅过去可得，星期几则可覆盖未来。官方接口要求数值类别代码和 NaN 缺失值。

公开实现对单 target 做逐序列的平滑 target encoding，类别 `c` 的值为 `(series_mean + sum_c) / (1 + count_c)`；统计来自 cutoff 前、context 窗口内前向填充后的 target，未见类别回退到序列均值。多 target 时没有单一编码对象，直接使用原始 numeric codes。它不是统一的类别 embedding 机制；多 target 时重编号是否改变结果值得验证。

历史编码使用整个可用 context，并非对每个历史时刻分别在线编码；这本身不等于泄漏预测区间的 label，但不能宣称该特征在每个历史位置都严格因果。

来源：[covariate_encoding.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/covariate_encoding.py)、[forecaster.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/forecaster.py)。

## 关键实验结果：相对 TimesFM-3 提升了吗？

下表是 **Mistral 官方 model card 报告值**，不是我们的独立复现。错误指标改善按 `(TimesFM-3 - Timestral) / TimesFM-3` 计算；skill score 使用绝对分差。

| Benchmark / metric | TimesFM-3 | Timestral-209M | 改善 |
| --- | --- | --- | --- |
| GIFT-Eval MASE ↓ | 0.667 | 0.658 | 误差降低 1.35% |
| GIFT-Eval CRPS ↓ | 0.456 | 0.446 | 误差降低 2.19% |
| TIME MASE ↓ | 0.640 | 0.626 | 误差降低 2.19% |
| TIME CRPS ↓ | 0.536 | 0.522 | 误差降低 2.61% |
| fev SQL skill ↑ | 48.99 | 50.39 | +1.40 分 |
| fev MASE skill ↑ | 37.82 | 39.49 | +1.67 分 |

Timestral-123M 在这六项上也均优于表中的 TimesFM-3，且接近自家 209M：GIFT MASE 为 0.659 vs 0.658，TIME MASE 为 0.628 vs 0.626。相对 t0-gamma，209M 的 GIFT MASE 是 0.658 vs 0.661，TIME CRPS 是 0.522 vs 0.534，fev SQL skill 是 50.39 vs 49.42。

来源：[官方比较表与各尺寸结果](https://huggingface.co/mistralai/Timestral-209M#benchmarks)。博客 GIFT 图标注 WQL，model card 表头写 CRPS；本文保留 model card 的原标签，不把它当成我们已重算的精确 CRPS 积分。

解释条件与证据边界：

- 博客的领先范围限定为 pretrained / zero-shot，来自单 checkpoint；没有独立确认实时榜首或统计显著性。
- 官方 README 说明使用默认 forecaster、最多 32,768 步历史；GIFT-Eval 为 97 configs 并输入 past covariates，fev 为 100 tasks 并使用历史/未来协变量和类别编码，TIME 为 98 tasks 且无协变量。
- 聚合分数没有展示逐任务收益分布、置信区间，也不能单独排除数据覆盖与预训练重叠问题。TIME 的提升说明优势不只出现在带协变量 benchmark，但仍不能归因于某个模块。
- 数据、context、训练预算、DHF、标准化和推理增强的贡献没有被现有材料逐项隔离。应称“一致的小幅整体改进”，而非已证明某个架构显著胜出。

来源：[官方 README 的 benchmark 设置](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/README.md#benchmarks)。

## GitHub / Code Analysis：官方实现

实际阅读路径：

```text
loading.py：配置和权重
  → TimestralForecaster.forecast
  → categorical encoding / _assemble_rows
  → _block_ar_decode
  → TimestralModel.predict_block
       ├─ _predict_context_only
       └─ _predict_window（带角色和 known-future）
  → masks + time/variate Transformer + QuantileHead
  → _refine_block / cap / symmetric average / clip
```

| 模块 | 实际职责 |
| --- | --- |
| `config.py` | forward / decode 配置，层顺序、role 支持检查 |
| `model.py` | patch embedding、双流组装、两个预测入口、单调分位数 head |
| `layers.py` | RMSNorm、SwiGLU、RoPE、双轴 attention、SDPA |
| `masks.py` | 时间 mask、group 隔离、变量 firewall、position IDs |
| `scaler.py` | 因果统计、asinh、反标准化与 advancing frames |
| `covariate_encoding.py` | 类别平滑 target encoding |
| `forecaster.py` | 输入校验、缺失处理、角色拼装、rollout 与后处理 |
| `loading.py` | 本地 / HF checkpoint 解析及严格加载 |

从实际代码可确认：

1. `forecast` 接受 NumPy / Tensor，target 可用 `(T,)`、`(B,T)`、`(B,V,T)`；target 非有限值标为 missing。填充值与 observation mask 分开，不能把 forward-fill 理解成原观测没有缺失。
2. 公开包是 standalone inference，主要依赖 PyTorch、NumPy、safetensors、huggingface_hub。官方仓库没有训练 / 合成数据生成和完整 benchmark runner；不能把 Forge 的适配能力写成本地现成 `fit` 功能。
3. `loading.py` 下载 `config.json` 与 `model.safetensors` 到 HF cache，也支持本地目录、revision、`local_files_only`。使用 `strict=True` 加载，默认 CPU / float32；推理不依赖远程预测 API。
4. 已读 `tests/test_forecaster.py`：覆盖输出形状、有限性/分位数顺序、协变量正负对称性、类别编码等价、加载路径。测试使用随机初始化小模型；它们不等于正式权重的精度、概率校准或 benchmark 复现。

来源：[loading.py](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/src/timestral/loading.py)、[tests](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/tests/test_forecaster.py)，其余源码见上文。

### License 边界

官方推理代码为 Apache-2.0，权重另用 Modified MIT。直接核对 209M 权重 LICENSE 后，必须记录其中的营收限制：若公司或雇主前一个月全球合并月营收超过 2,000 万美元（或等值货币），该许可不授予使用权；条款也覆盖衍生、修改与组合产物，并提供另行申请商业许可的途径。不能只看到 “MIT” 就记成无限制商用。

第三方 extension 的 MIT 也不会消除其 ONNX 权重的限制。来源：[官方权重 LICENSE](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)、[官方代码 LICENSE](https://github.com/mistralai/timestral/blob/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff/LICENSE)。

## 第三方 DuckDB 扩展：移植了什么，为什么没做协变量？

`altertable-ai/duckdb-timestral` 不是 Mistral 官方仓库。实际执行路径是：

```text
SQL 分组 / 时间网格
  → C++ 预处理
  → ONNX Runtime CPU：单块预测图
  → C++ refine / rollout / cap / symmetric average / clip
  → SQL 中位数与 21 个分位数
```

已读 `scripts/export/export_onnx.py`、`src/ort_predictor.cpp`、`src/forecaster.cpp` 和设计文档。导出脚本封装官方 `_predict_context_only`，模型 revisions 固定；C++ 移植外层预测流程。运行时不嵌 Python；ORT 单次执行设置单线程，README 说明跨序列并行交给 DuckDB worker threads。

| 能力 | 当前 extension |
| --- | --- |
| 单变量、按组分别预测 | 支持 |
| 21 个分位数、缺失值和时间网格处理 | 支持 |
| 多 target 联合预测 | 未支持 |
| Past-only / known-future / categorical covariates | 未支持 |

示例（15 分钟电价，预测 96 步）：

```sql
SELECT region, ts, forecast, q10, q90
FROM electricity_prices TIMESTRAL_FORECAST (
    price
    PARTITION BY region
    ORDER BY ts
    HORIZON 96
    EVERY INTERVAL '15 minutes'
    USING MODEL '123m'
);
```

`PARTITION BY region` 是各区域独立预测，不会产生区域间 variate attention。此例是接口说明，未实际运行。

### 缺少协变量是范围选择，尚无技术不可行证据

设计文档 §3 明确把 covariates 和 multivariate targets 排除在 v1 之外，且标记该范围选择由构建 agent Claude 作出；§6.4 提出将来扩展 SQL 参数、用未来行携带 known-future 信息，并把 `_predict_window` 另导出为图。这是仓库里的设计记录，不代表维护者已承诺 v2 交付时间。

| 要扩展的层 | 当前实现 | 协变量需要增加的工作 |
| --- | --- | --- |
| SQL contract | 每组 time + 一个 value | 声明多个角色与类别列，区分历史与预测窗口 |
| 时间网格 | 根据 target 行构造序列与预测起点 | 对齐未来协变量，定义未来 NULL target 行的语义 |
| ONNX export | Context-only 图 | role IDs、known-future values / masks 与角色相关路径 |
| C++ rollout | 回填目标预测 | 分别处理 target 中位数、known-future 已知值、past-only 缺失 |
| 类别与测试 | 无类别路径 | 编码、未见类别、缺失值、与官方 PyTorch 的一致性验证 |

**我们的推断**：先完成分组单变量预测，能收窄 SQL 语义和数值一致性验证的范围；协变量需要端到端扩展，远多于增加几个 SQL 参数。没有证据表明 DuckDB 或 ONNX 从根本上无法实现。

工程上还要注意：首次使用会从 Altertable 的 asset server 下载 ONNX 并做 SHA-256 校验，缓存至 `~/.duckdb/timestral/v1/`；可指定本地模型离线使用。README 说明模型在 query planning 阶段加载，因此 `EXPLAIN` 也可能触发下载。项目声称 parity tests 对比官方全部 21 quantiles，我们未运行，不能将其数值误差声明当成独立复核结果；README 的精度栏引用 Mistral 成绩，也不等于 extension 在完整协变量 benchmark 上复现了该成绩。

来源：[扩展 README](https://github.com/altertable-ai/duckdb-timestral/blob/6936c3957551ec6df255b19c47b6cf822b05efb8/README.md)、[导出脚本](https://github.com/altertable-ai/duckdb-timestral/blob/6936c3957551ec6df255b19c47b6cf822b05efb8/scripts/export/export_onnx.py)、[ORT predictor](https://github.com/altertable-ai/duckdb-timestral/blob/6936c3957551ec6df255b19c47b6cf822b05efb8/src/ort_predictor.cpp)、[C++ forecaster](https://github.com/altertable-ai/duckdb-timestral/blob/6936c3957551ec6df255b19c47b6cf822b05efb8/src/forecaster.cpp)、[设计 §3 / §6.4](https://github.com/altertable-ai/duckdb-timestral/blob/6936c3957551ec6df255b19c47b6cf822b05efb8/docs/superpowers/specs/2026-10-20-timestral-extension-design.md)。

## 关键分析

- **结构贡献有可读实现，性能归因仍缺证据。** 双流 forecast queries、角色 firewall 和非交叉 head 都能定位到源码；没有受控 ablation 就不能把几个百分点的整体领先分配给它们。
- **Synthetic-first 的潜在研究价值很高，公开缺口也大。** 我们关心的是多变量依赖如何生成、覆盖哪些机制及 shift，而不只是合成数据占比；现有推理代码回答不了。
- **完整推理配方会影响结论。** 双次对称预测、非负裁剪、长 context 与缩放策略都可能贡献效果，应与裸 backbone 分开评估。
- **原生模型能力与集成入口是两层事实。** 官方支持 future covariates，不代表 DuckDB 接口已支持；扩展的单变量 benchmark parity 也不能替代协变量任务验证。

## Insights

1. **变量角色由任务与 cutoff 决定。** 历史电价是 target context，历史出力可以是辅助观测，也可以换成另一个 target；预测产品与真实观测必须分成可复核的数据来源。
2. **时间泄漏是跨模块路径问题。** 只给 target 加 causal mask 不够；双向协变量可能成为中继，类别编码和上游数据版本也可能改变可得性边界。
3. **未来信息的价值来自“提前可用”，不要求“无误差”。** 预报 archive 的 issue-time contract 与模型的 mask 同样关键。
4. **联合表示、边际概率与联合场景是不同能力。** 跨变量 attention 不自动提供有正确相关性的风险场景。
5. **模型部署便利与方法完整性需要分别评价。** SQL 内预测能显著降低数据搬运成本，但对电价等强协变量任务，省略协变量会改变问题本身。

## 我们的观点

这次发布值得继续看，尤其是 synthetic multivariate prior 与角色约束的信息流。相对 TimesFM-3，现阶段最稳妥的评价是“一套更强的整体配方，在公布的聚合指标上小幅一致领先”；不足以宣称 DHF 已被证明是决定性突破。

对我们的电价问题，优先级应放在：能否正确读取气象 / 新能源 / 负荷预测版本，以及它们对不同市场阶段的增益，而不只是复用零样本榜单。应比较 target-only、加历史实际值、再加预测期协变量的增量，并检查非负裁剪对负价风险的影响。

DuckDB 扩展是有实际工程内容的部署工作，适合作为 SQL 内单变量 baseline。当前拿它替代官方完整条件预测接口，会丢掉我们最关心的一部分能力。

## 值得继续追的问题

- DHF vs CPM 在固定数据、预算、context、后处理下，净收益多少？训练目标与真实缺失增强的解耦能否单独验证？
- 合成单变量过程、多变量 nonlinear dependency 和真实数据各贡献多少？训练集与 benchmark 的重叠检查如何做？
- 电价预测中，past-only actuals、known-future forecasts 的增量分别多大？预报误差或发布延迟变化时是否稳定？
- 关闭 symmetric averaging、non-negative clipping、refine 后，准确率与校准怎样变化？
- 长 horizon 的中位数回喂是否低估不确定性？联合变量尾部事件能否用显式场景评估？
- 多 target 的 categorical numeric codes 对重新编号是否敏感？需要怎样的编码才能保留类别语义？
- 123M 与 209M 在真实任务上的精度、延迟、变量数 scaling 如何权衡？
- DuckDB 若补协变量，如何定义历史 / 未来 relation、forecast origin、预报版本与缺失策略，并验证角色 mask 的数值一致性？
