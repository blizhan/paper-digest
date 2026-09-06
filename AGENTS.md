# AGENTS.md

本仓库是论文阅读与技术判断的长期记录，不是 paper dump。后续 agent 在这里工作时，目标是把“我们真正读过并讨论过的论文”沉淀成可复用的研究笔记，并保持论文事实、代码事实和主观判断之间的边界清楚。

## Scope

- 只收录用户已经实际读到、讨论到，或明确要求加入仓库的论文。
- 不要因为论文出现在搜索结果、推荐列表、浏览器标签页或 related work 里就自动建目录。
- 如果用户明确说某篇“还没读到”“先不加”，不要创建占位条目，也不要把它加入根 README。
- 一篇论文对应 `knowledge/papers/` 下一个独立 Markdown concept，文件名使用英文标题的可读 kebab-case slug。
- 论文 concept 必须带 OKF 风格 YAML frontmatter，至少包含非空 `type`，建议同时维护 `title`、`description`、`resource`、`tags` 和 `sources`。
- 根 `README.md` 是面向人的仓库入口；`knowledge/index.md` 是 OKF bundle 根索引；`knowledge/papers/index.md` 是论文层的 progressive-disclosure 索引。长篇分析只放在论文 concept 文档中。

## OKF 目录约定

本仓库参考 Open Knowledge Format v0.2，但把 bundle 放在 `knowledge/` 子目录中，使根目录可以继续保留 `README.md`、`AGENTS.md` 等仓库级文件而不把它们当作 OKF concept。

```text
papers/
├── README.md
├── AGENTS.md
└── knowledge/
    ├── index.md
    └── papers/
        ├── index.md
        ├── <paper-slug>.md
        └── ...
```

- `knowledge/index.md` 使用 `okf_version: "0.2"`，只负责暴露下一层目录。
- `knowledge/papers/index.md` 不使用 frontmatter，按 OKF `index.md` 规则列出论文 concept 和对应的简短 description。
- `knowledge/papers/*.md` 是 concept 文档，必须以 YAML frontmatter 开头。
- 新增其他知识类型时优先新建同级子目录和对应 `index.md`，不要把所有内容都堆进 `papers/`。

## 每篇论文最低记录要求

单篇笔记至少应包含下面这些内容；可以按论文类型调整章节顺序，但不要遗漏核心信息：

1. `Links`
   - arXiv / publisher / Hugging Face Papers 等论文入口；
   - 官方项目页；
   - 官方 GitHub / code repository（如果存在）。
2. `一句话结论`
   - 用 1–2 段说明“这篇真正做了什么、值不值得继续看”。
3. `文章摘要`
   - 用自己的话概括 problem、method、evaluation 和主要 conclusion；
   - 不要直接大段复制 abstract。
4. `方法拆解 / 这篇到底做了什么`
   - 把系统、模型、数据处理、训练和推理路径拆开；
   - 对容易被标题或 marketing 混淆的部分给出更准确的技术描述。
5. `关键实验结果`
   - 记录真正支撑判断的数字和 baseline；
   - 注明诸如 single-seed、subset、API-only、system submission 等会影响解释的条件。
6. `关键分析`
   - 区分主要贡献、增量改进、工程工作和 empirical finding；
   - 指出论文 claim 的证据强弱和可能的 confounder。
7. `Insights`
   - 保留以后看同类论文仍有用的抽象结论，而不是重复论文摘要。
8. `我们的观点`
   - 明确这是讨论后的主观判断；
   - 可以直接评价工程味、营销成分、研究价值和优先级，但必须有前文事实支撑。
9. `GitHub / Code Analysis`（有公开代码时）
   - 分析仓库结构、核心实现路径、关键模块、配置与依赖；
   - 对照论文确认“论文说的”和“代码真正做的”是否一致；
   - 记录复现限制、license、API dependency、cache、数据预处理等工程事实。
10. `值得继续追的问题`
    - 留下 ablation、benchmark coverage、scaling、deployment 或后续论文值得验证的问题。

## 事实、推断与观点的边界

- **论文事实**：来自论文正文、官方 README、官方 release/result 等一手资料。写成确定语气前必须能定位来源。
- **代码事实**：必须来自实际阅读官方代码，而不是仅根据 README 或论文猜实现。
- **我们的推断**：例如“创新主要发生在 pipeline 层”，可以写，但应说明这是基于哪些实现/实验得出的判断。
- **我们的观点**：例如“工程贡献多于算法创新”“有明显自家模型 positioning”，可以直接写，但不要伪装成作者 claim。
- 对不确定信息使用“目前看”“从公开实现看”“官方仓库当前版本中”等限定词，不要补全未知细节。

## 阅读与来源优先级

优先级建议：

1. 论文 PDF / arXiv 正文；
2. 官方 GitHub 与具体源码；
3. 官方 project page / model card / Hugging Face paper page；
4. 作者公开说明；
5. 第三方解读仅用于补充，不应用来覆盖一手资料。

如果浏览器里打开的 arXiv 编号、旧版本或其他页面与官方仓库引用的正式论文不一致，先核对再落笔。根 README 与单篇笔记应使用已经确认的 canonical paper link。

## GitHub / Code Analysis 工作方式

- 不要只看仓库首页就写“代码分析”。至少定位到承载核心方法的源码文件。
- 推荐顺序：
  1. 阅读仓库 README 和 paper/release 对应说明；
  2. 找 model entrypoint / registry / config；
  3. 沿 `fit` / `predict` / data preprocessing 的真实调用链阅读；
  4. 检查 evaluation、split、cache、API/client、feature construction 等会影响结论的代码；
  5. 必要时读 tests / examples 验证语义。
- 如果临时 clone 官方仓库用于分析，优先放在 `/private/tmp`，不要把第三方源码 vendoring 到本仓库。
- 在笔记里记录实际分析时对应的 commit SHA，避免未来仓库变化后无法复核。
- 若本地/hosted 两条实现路径不同，要明确拆开；不要把 API 能力误写成本地 OSS 能力。

## 写作风格

- 中文为主，技术名词保留常用英文，例如 `target grain`、`cutoff`、`DFS`、`context selection`。
- 直接、技术化，不需要论文式套话。
- 优先解释“为什么”，尤其是 prediction grain、temporal leakage、baseline fairness 这类容易被忽略的条件。
- 对复杂 pipeline 可以使用短代码块或 ASCII flow；避免为了格式而格式化。
- 数字、排名、模型名称尽量保留原始精度和实验条件。
- 不要为了显得完整而增加尚未验证的内容。

## 索引维护规则

每新增一篇已经完成讨论的论文：

- 在根 `README.md` 的索引表增加一行；
- 在 `knowledge/papers/index.md` 增加对应 concept 链接和与 frontmatter `description` 一致的短描述；
- 至少包含 paper 名称、topic、主要 links、阅读状态和一句话判断；
- 索引的一句话判断要能和单篇笔记中的结论对应；
- 不创建“待阅读”大列表，除非用户明确要求仓库承担 reading queue 的功能。

## 验证清单

完成一次新增或修改后，至少检查：

```bash
rg --files | sort
git diff --check
git status --short
```

另外人工确认：

- 新论文是否真的已经讨论过；
- 根 `README.md`、`knowledge/papers/index.md` 与论文 concept 链接是否一致；
- `knowledge/papers/*.md` 是否都包含可解析 YAML frontmatter 和非空 `type`；
- arXiv / GitHub 链接是否指向正确版本；
- 是否把推断写成了事实；
- 有代码时是否真的读了核心实现；
- 是否误加入用户明确要求暂不收录的论文。

## Git 行为

- 默认只修改工作区，不自动 commit、push、开 PR。
- 保留用户已有修改；不要为了整理仓库覆盖无关内容。
- 用户要求 commit 时，再根据当前变更组织清晰的 commit message。

## 当前仓库约定的一个参考范例

`knowledge/papers/advancing-open-and-reproducible-relational-learning-relarena-tabpfn-rel-rpi.md` 可以作为后续单篇笔记的结构参考：它把论文摘要、方法拆解、prediction grain、实验结果、Insights、我们的观点和实际源码分析分开记录。

后续不要求所有论文机械复制同一模板；如果论文是纯 benchmark、dataset、system、theory 或 architecture paper，可以调整结构，但“事实可复核 + insight 可复用 + 主观判断有依据”这三个目标保持不变。
