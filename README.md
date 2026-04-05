# Metagen

这是一个面向自由形态二值超表面的逆向设计项目。项目的核心目标不是只优化某个单点透过率，而是设计在给定波长和入射角范围内具有指定角谱响应的 Fourier-operator metasurface，也就是让超表面的光学传递函数在角度-波长空间中呈现出目标形状。

从物理上看，这类器件希望直接在光学层面实现微分、滤波、偏振复用、时空微分等功能；从计算上看，这又是一个高维、强非线性、一对多且优解稀疏的逆问题。这个仓库围绕这两个层面，建立了一条相对完整的研究闭环：

- 用自由形态结构生成与 RCWA 仿真构建 `structure -> angle-resolved OTF` 数据
- 用前向代理模型学习结构到响应的近似映射，降低全 RCWA 搜索成本
- 用条件扩散模型学习从目标响应回到结构分布的逆映射
- 用任务化的 RCWA 重评分与可视化流程验证不同光学功能目标

当前最重要的研究入口之一是 `src/infer/run_diffusion_tasks.py`。它把“任务定义 -> 条件生成 -> RCWA 重评分 -> 结果汇总”串成了一条统一流程，也是当前仓库最能体现课题主线的脚本。

## 项目在做什么

这个项目关心的是 OTF engineering，也就是通过设计超表面单元的空间结构，让器件在傅里叶空间中呈现出目标传递函数。对于二阶微分器、高阶微分器、低通滤波器或偏振复用器件来说，真正关键的并不是某个单独角度的透过率，而是整个角谱响应在目标工作区间内的形状。

这也是本项目为什么一直围绕 `lambda-theta` 响应来建模，而不是只做单波长单角度优化。项目更关心的是：

- 结构如何决定角分辨光谱响应
- 不同偏振通道如何共同决定器件功能
- 如何把不同物理目标统一表述为“目标响应 + 评分规则”
- 如何在 RCWA 成本很高的前提下，仍然完成可用的逆向设计闭环

## 当前目录

```text
src/
├── dataset/
│   ├── materials/      # 材料折射率数据与插值
│   ├── rcwa/           # TORCWA 仿真与批量数据生成
│   └── structure/      # 64x64 自由形态结构生成
├── model/
│   ├── dataset.py      # 训练数据读取
│   ├── models.py       # 前向代理 + 条件 UNet
│   ├── diffusion.py    # 扩散过程
│   ├── train_utils.py  # 训练日志与可视化工具
│   ├── train_forward.py
│   ├── train_diffusion.py
│   └── sample.py
├── infer/
│   ├── common.py               # 模型加载、RCWA 工具、可视化公共函数
│   ├── task_library.py         # 任务清单、case 选择、任务评分逻辑
│   ├── run_diffusion_tasks.py  # 任务驱动逆设计主入口
│   ├── validate_surrogate_top.py
│   ├── imag_process/           # 傅里叶光学成像验证
│   ├── st/                     # ST 相关输入构造、分析、虚拟实验与复现脚本
│   └── laplas.py / optimization.py  # 历史参考脚本
└── baselines/
    ├── model/
    ├── train/
    └── eval/
```

仓库根目录下还包括：

- `data/`：评分、筛选、可视化、对比分析脚本与对应输出目录
- `checkpoints/`：前向代理与扩散模型权重、统计量、训练日志
- `samples/`：采样、任务推理、评估对比等结果目录
- `runs/`：TensorBoard 日志
- `papers/`：论文、笔记与 ST 相关资料

各模块在课题中的角色大致是：

- `src/dataset`
  - 负责生成自由形态结构，并通过 RCWA 建立“结构 -> 角分辨 OTF 响应”的数据基础
- `src/model`
  - 负责学习正向映射和逆向生成映射
- `src/infer`
  - 负责把不同物理功能组织成任务，并执行逆设计推理与 RCWA 验证
- `src/infer/st`
  - 负责时空微分相关分析与扩展实验
- `data`
  - 负责评分、筛选和数据分析，是项目里的“物理目标量化层”

## 推荐阅读顺序

- 如果你想先理解项目的物理主线，先看本 README 的“物理问题与数据表示”和“任务驱动逆设计”
- 如果你想了解数据怎么来，先看 `src/dataset/structure` 和 `src/dataset/rcwa`
- 如果你想了解模型训练，先看 `src/model/train_forward.py` 和 `src/model/train_diffusion.py`
- 如果你想直接看当前主线逆设计流程，先看 `src/infer/task_library.py` 和 `src/infer/run_diffusion_tasks.py`
- 如果你想看评分和候选筛选怎么定义，集中看 `data/`
- 如果你想看 ST 相关扩展实验，集中看 `src/infer/st`

## 物理问题与数据表示

本项目里最核心的对象不是“结构图像”本身，而是结构对应的角分辨 OTF 响应。对一个给定超表面来说，我们更关心的是它在不同波长 `lambda` 和不同入射角 `theta` 下，对不同偏振通道的透射响应如何变化。这也是为什么项目数据不是简单的标量标签，而是二维响应图。

当前数据集主要使用两个通道：

- `tpp_mag`
- `tss_mag`

它们对应不同偏振通道下的幅值响应。代码中大部分任务都围绕这两个通道展开，因为很多目标功能本质上就体现为：

- 某个目标波长处，一条角度曲线的形状
- 或者一个有限 `lambda-theta` 窗口中的二维模式

这也解释了为什么：

- 二阶、高阶、低通等任务更关注目标波长处的一维角谱曲线
- 偏振相关任务需要联合看多个通道
- `st2` 任务更像二维窗口目标，而不是单一波长的单条曲线

这些物理量在项目中的落点是：

- `src/dataset/rcwa/rcwa_all.py`
  - 把结构样本标注为角分辨响应
- `train_data.npz`
  - 保存结构与 `tpp_mag/tss_mag` 的配对数据
- `src/model/dataset.py`
  - 把这些物理响应整理成模型训练使用的条件输入
- `src/infer/task_library.py`
  - 把不同器件功能翻译成任务 case 与评分规则

## 数据流程

### 1. 结构生成

`src/dataset/structure/dataset_pre.py`

项目当前使用的是 `64x64` 自由形态二值结构。这样的表示方式比少量几何参数更自由，也更接近“通用逆设计”的设定，但同时也使搜索空间大幅增加，所以需要后续的数据驱动建模来降低搜索难度。

- 默认生成 `5000` 个 `64x64` 二值结构，也可通过参数传入任意样本数
- 每个样本使用不同随机种子
- 结构满足 `C4 + sigma_x` 对称
- 约定 `1 = 材料`，`0 = 空气`
- 批量生成时固定使用 `generate_one.py` 当前参数：
  - `N_COARSE = 8`
  - `N_FINE = 64`
  - `SIGMA1 = 1.9`
  - `SIGMA2 = 1.1`
  - `MIN_FEATURE_PX = 7`
- `TARGET_FILL` 在 `0.4 / 0.5 / 0.6 / 0.7 / 0.8` 五档之间按样本数尽量平均分配
- 输出到 `data/structures/structures.npy`

运行：

```bash
python src/dataset/structure/dataset_pre.py
```

如果要指定样本数：

```bash
python src/dataset/structure/dataset_pre.py --num_samples 5000
```

### 2. 数据增强（可选）

`src/dataset/augment_dataset.py`

由于全 RCWA 标注成本较高，项目使用前向代理模型先对大批量结构做快速打分，再筛选出更有价值的子集进入 RCWA。它的研究意义不是简单“加速”，而是提升训练数据中优质样本比例，让后续逆设计更容易学到有效结构。

前置条件：需要已有 `checkpoints/forward_best.pt` 和 `checkpoints/cond_stats.npz`。

```bash
python src/dataset/augment_dataset.py --num_pool 100000 --topk 2000 --num_random 1000
```

跳过生成、直接筛选：

```bash
python src/dataset/augment_dataset.py --skip_generate --topk 2000 --num_random 1000
```

输出：

- `data/structures/structures_pool.npy`
- `data/structures/structures_augmented.npy`
- `data/structures/augmented_scores.npy`

### 3. RCWA 批量仿真

`src/dataset/rcwa/rcwa_all.py`

这一步负责建立项目最基础的“结构 -> 角分辨 OTF”配对数据。RCWA 在这里不是可选的可视化工具，而是整个项目的真实物理标注来源，也是后续推理阶段的最终评测口径。

固定物理参数：

- 周期 `500 nm`
- 厚度 `500 nm`
- 结构材料 `Si`
- 入射介质 `air`
- 输出介质 `SiO2`
- `phi = 0`
- `angle_layer = input`

采样网格：

- `lambda = 800, 850, ..., 1300`，共 `11` 个点
- `theta = -40, -35, ..., 40`，共 `17` 个点

默认输出到 `data/train_data.npz`，并额外写：

- `data/rcwa.log`
- `data/train_data_failures.json`

运行：

```bash
python src/dataset/rcwa/rcwa_all.py
```

如果要修改样本数或阶数：

```bash
python src/dataset/rcwa/rcwa_all.py --max_samples 100 --rcwa_orders 7
```

如果要多卡并行：

```bash
python src/dataset/rcwa/rcwa_all.py --devices cuda:0,cuda:1 --max_samples 5000 --rcwa_orders 7
```

### 4. 训练数据格式

`train_data.npz` 当前约定如下：

- `structures`: `[N, 64, 64]`
- `tpp_mag`: `[N, 11, 17]`
- `tss_mag`: `[N, 11, 17]`
- `lambdas`: `[11]`
- `thetas`: `[17]`

其中二维响应图固定按 `[lambda, theta]` 排列。

`src/model/dataset.py` 会把它读成：

- `x`: `[1, 64, 64]`
- `cond`: `[2, 11, 17]`

这里 `2` 个通道分别是 `tpp_mag` 和 `tss_mag`。这意味着模型学习的不是单值标签，而是目标 OTF 响应图到结构分布的关系。

## 模型部分

### 前向代理模型

`src/model/train_forward.py`

前向代理模型的作用是学习“结构 -> 响应”的近似映射。它在项目里不是边角料，而是一个重要的中间层：既能用于数据增强和候选预筛，也能作为训练期物理一致性约束的基础。

- 输入结构 `[1, 64, 64]`
- 预测条件图 `[C, 11, 17]`
- 当前损失是 `L1`
- 训练日志由 `src/model/train_utils.py` 统一管理

运行：

```bash
python src/model/train_forward.py
```

输出：

- `checkpoints/forward_last.pt`
- `checkpoints/forward_best.pt`
- `checkpoints/cond_stats.npz`
- `checkpoints/forward_log.csv`
- `runs/forward`

### 条件扩散模型

`src/model/train_diffusion.py`

逆向设计本质上是一对多问题：同一个目标 OTF 可能对应多种候选结构。因此这里采用条件扩散模型，而不是把问题写成单值回归。它承担的是“从目标响应反推结构分布”的角色。

- 用条件图反推结构
- 当前训练损失由扩散主损失、训练期物理一致性损失和二值正则共同组成
- 输入结构训练时会从 `[0, 1]` 映射到 `[-1, 1]`
- 训练日志与预览样本由 `src/model/train_utils.py` 统一管理

运行：

```bash
python src/model/train_diffusion.py
```

输出：

- `checkpoints/diffusion_last.pt`
- `checkpoints/diffusion_best.pt`
- `checkpoints/diffusion_log.csv`
- `checkpoints/diffusion_preview/*.npy`
- `runs/diffusion`

### 基础采样

`src/model/sample.py`

这是一个较基础的条件采样入口，用于从目标条件图生成多个候选结构，再由前向代理做快速排序。

```bash
python src/model/sample.py
```

## 任务驱动逆设计

当前推荐的逆设计主线是 `src/infer/run_diffusion_tasks.py`。

它的基本流程是：

1. 从任务库中选定目标 case
2. 以目标响应作为条件生成候选结构
3. 用 RCWA 回到真实物理域重评分
4. 输出结构图、响应图、曲线图和 `summary.json`

这条流程的重要性在于，它把“逆设计能力”从抽象的生成误差，变成了“在具体器件功能目标上是否能得到高分结构”的问题。项目中的不同任务不是零散脚本，而是对不同 OTF 工程目标的统一表达。

### 六类任务

#### 1. `p_second_order`

- 二阶空间微分器任务
- 关注目标波长处 `tpp` 角谱曲线是否接近二阶型响应
- 是项目最基础、最核心的验证任务

#### 2. `polarization_independent`

- 偏振无关功能任务
- 关注不同偏振通道是否都能维持相近目标响应
- 体现结构在多偏振条件下的稳健性

#### 3. `polarization_multiplexed`

- 偏振复用功能任务
- 关注不同偏振承担不同响应时的区分度
- 体现一个结构承载多功能的能力

#### 4. `fourth_order`

- 高阶微分任务
- 关注更尖锐的高阶目标曲线拟合
- 体现模型和设计流程对更复杂目标的处理能力

#### 5. `lowpass`

- 低通滤波型 OTF 任务
- 关注中心角高透过、边缘角抑制的低通响应
- 说明项目不仅做微分算子，也能做滤波型目标

#### 6. `st2`

- 时空微分相关任务
- 关注二维 `lambda-theta` 窗口内的模式匹配
- 说明项目可以从单波长曲线目标扩展到更复杂的二维目标

具体 case 清单、样本编号和评分细节以 `src/infer/task_library.py` 为准，README 不把全部 case 表硬编码进去。

### 运行方式

跑一类任务：

```bash
python src/infer/run_diffusion_tasks.py --tasks p_second_order --num_samples 16 --topk 5 --eval_mode target_only
```

精确跑一个 case：

```bash
python src/infer/run_diffusion_tasks.py --tasks polarization_multiplexed:1100nm_id11339 --num_samples 16 --topk 5 --eval_mode full
```

### 输出内容

`run_diffusion_tasks.py` 会按 `task / case` 组织结果目录，重点输出包括：

- 目标结构与目标条件
- 所有候选结构与评分
- top-k 对比图
- 目标波长曲线图与总览图
- `summary.json`

## data 目录脚本

`data/` 在这个项目里不是辅助脚本仓库，而是任务定义、候选筛选与样本分析的重要组成部分。这里的脚本承担的是把不同物理功能量化成 score 的工作，因此它可以被看作项目里的“物理目标量化层”。

按用途大致可以分为：

- 微分类 / 高阶类：
  - `score_second_order.py`
  - `score_fourth_order.py`
- 偏振相关类：
  - `score_polarization_independent.py`
  - `score_polarization_multiplexed.py`
  - `score_polarization_multiplexed_hightrans.py`
- 滤波类：
  - `score_optica_highpass.py`
  - `score_optica_lowpass.py`
  - `score_optica_lowpass_selective.py`
- ST 类：
  - `score_st2_sample.py`
  - `screen_st2_candidates.py`
  - `screen_st2_template_match.py`
  - `filter_st_window_candidates.py`
- 筛选与对比类：
  - `filter_lambda0_window_candidates.py`
  - `visualize_dataset.py`
  - `plot_sample_tpp_grid.py`
  - `compare_sample_to_st2_ideal.py`
  - `check_polarization_consistency.py`

这些脚本大多直接消费 `data/train_data.npz`，并把结果写到各自的 `data/*_scores` 或分析输出目录中。

常用示例：

```bash
python data/visualize_dataset.py
python data/score_second_order.py
python data/score_optica_lowpass.py
```

## ST 相关脚本

`src/infer/st/` 汇总了 ST 相关的输入构造、理想响应分析、样本对比、虚拟实验和论文风格绘图脚本。

这一部分目前更偏研究分析与扩展实验，不是项目的最小主线；如果你关心 ST 任务，建议优先结合 `src/infer/st/common.py`、`src/infer/st/build_inputs.py`、`src/infer/st/run_virtual_experiments.py` 阅读。

## Baseline 对比

`src/baselines/` 提供了若干对比方法，用来回答“为什么当前这套生成式逆设计流程值得采用”这个问题。目前主要包括：

- `topo_opt`
- `cvae`
- `cgan`
- `diffusion`

训练示例：

```bash
python src/baselines/train/train_cvae.py --data_path data/train_data_20000.npz --save_dir checkpoints/cvae
python src/baselines/train/train_cgan.py --data_path data/train_data_20000.npz --save_dir checkpoints/cgan
```

评估示例：

```bash
python src/baselines/eval/run_eval.py --task_case p_second_order:1050nm_id4279
```

## 成像验证

`src/infer/imag_process/` 用于把设计得到的传递函数重新放回傅里叶光学成像语境中做补充验证。

最基础的示例：

```bash
python src/infer/imag_process/image_processing.py --ideal
python src/infer/imag_process/scan_kspace.py --sample 0
```

## 环境准备

推荐使用 `conda` 单独创建环境。

### 1. 拉取项目

```bash
git clone https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

如果服务器访问 GitHub 较慢，也可以在原始链接前加加速前缀：

```bash
git clone https://ghfast.top/https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

### 2. 创建环境

```bash
conda create -n metagen python=3.10 -y
conda activate metagen
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

依赖说明：

- 想跑数据链与模型：需要 `torch`
- 想跑 RCWA 标注和物理评估：需要 `torcwa`
- 想跑绘图和成像分析：需要 `matplotlib`、`Pillow`
- 基础数据处理依赖 `numpy`、`scipy`

建议始终在仓库根目录执行命令，因为大多数脚本默认使用 `data/...`、`checkpoints/...`、`samples/...` 这样的相对路径。

## 当前已知问题

1. `pyproject.toml` 目前只有最小项目信息，没有完整声明运行依赖。
2. 许多脚本默认依赖仓库根目录下的相对路径，离开根目录执行时容易找不到文件。
3. RCWA 批量仿真计算量较大，目前只有日志和失败记录，没有完整断点续跑。
4. `train_data.npz` 中失败点可能以 `NaN` 保存，训练和评分前需要自行确认是否过滤。
5. 某些环境下 `matplotlib/fontconfig` 缓存目录不可写会报警，但不一定影响脚本逻辑。

## 最小运行链

完整训练链：

```bash
python src/dataset/structure/dataset_pre.py --num_samples 5000
python src/dataset/rcwa/rcwa_all.py
python src/model/train_forward.py
python src/model/train_diffusion.py
python src/infer/run_diffusion_tasks.py --tasks p_second_order
```

如果已经有 `train_data.npz` 和 checkpoint，更常见的研究分析链是：

```bash
python data/score_second_order.py
python data/score_optica_lowpass.py
python src/infer/run_diffusion_tasks.py --tasks lowpass
```

如果环境支持 TensorBoard，可用下面命令查看训练曲线：

```bash
tensorboard --logdir runs
```
