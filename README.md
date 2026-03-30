# Metagen

面向超表面逆向设计的研究型原型。当前仓库已经形成一条最小数据链：

`自由形态结构生成 -> RCWA 仿真 -> train_data.npz -> 前向代理模型 -> 条件扩散逆向设计`

现在的代码主线在 `src/dataset` 和 `src/model`，旧版 `metaedge_ai` 骨架说明已经废弃。

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
│   ├── common.py                  # 公共工具：模型加载 / 可视化
│   ├── task_library.py            # 六类任务清单 + 任务评分公式
│   ├── run_diffusion_tasks.py     # 扩散生成 + RCWA 任务打分 + 可视化
│   ├── optimize_task_candidates.py # top-k 候选的 RCWA 拓扑优化
│   ├── validate_surrogate_top.py  # 代理模型验证
│   ├── imag_process/              # 傅里叶光学成像仿真（通用版，支持任意 --lambda_nm）
│   │   ├── image_processing.py
│   │   └── scan_kspace.py
│   └── optimization.py            # 旧版单目标优化脚本（保留作历史参考）
└── baselines/
    ├── model/
    │   ├── cvae.py     # CVAE 架构
    │   └── cgan.py     # cGAN 架构（Hinge loss + projection discriminator）
    ├── train/
    │   ├── train_cvae.py
    │   └── train_cgan.py
    └── eval/
        ├── metrics.py   # best_score / success_rate / spectrum_mae / diversity
        ├── infer_all.py # 统一生成接口（random / CVAE / cGAN / diffusion）
        ├── run_eval.py  # 跑完整对比实验 → results.json
        └── plot_results.py  # 绘制 4 张对比图
```

## 数据流程

### 1. 结构生成

`src/dataset/structure/dataset_pre.py`

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

### 1.5 数据增强（可选）

`src/dataset/augment_dataset.py`

用代理模型（ForwardSurrogate）对大批量结构快速打分，筛出高质量子集，再做 RCWA，可以显著提升训练数据中高二阶分数样本的比例，改善代理模型和扩散模型的泛化能力。

**策略（方案 B）**：top-K 高分 + num_random 随机样本混合，既保证质量又保留分布覆盖。

前置条件：需要已训练好的 `checkpoints/forward_best.pt` 和 `checkpoints/cond_stats.npz`。

运行（全流程：生成 10w + 筛选 top2000 + 随机 1000）：

```bash
python src/dataset/augment_dataset.py --num_pool 100000 --topk 2000 --num_random 1000
```

跳过生成（已有结构池时直接筛选）：

```bash
python src/dataset/augment_dataset.py --skip_generate --topk 2000 --num_random 1000
```

输出：

- `data/structures/structures_pool.npy`：生成的结构池
- `data/structures/structures_augmented.npy`：筛选后的训练子集
- `data/structures/augmented_scores.npy`：全量打分结果

然后对筛出的结构跑 RCWA：

```bash
python src/dataset/rcwa/rcwa_all.py --structures data/structures/structures_augmented.npy
```

完整增强 + 重训流程：

```bash
# 第一步：生成 10w + 代理筛选（~30min 生成 + 几分钟推理）
python src/dataset/augment_dataset.py --num_pool 100000 --topk 2000 --num_random 1000

# 第二步：只对筛出的 3000 个结构跑 RCWA
python src/dataset/rcwa/rcwa_all.py --structures data/structures/structures_augmented.npy

# 第三步：重训两个模型
python src/model/train_forward.py
python src/model/train_diffusion.py
```

### 2. RCWA 批量仿真

`src/dataset/rcwa/rcwa_all.py`

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

默认取前 `5000` 个结构；如果显式传 `--max_samples` 会按传入值截断。默认 `rcwa_orders=7`。输出到 `data/train_data.npz`，并额外写：

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

如果要多卡并行（按结构分片）：

```bash
python src/dataset/rcwa/rcwa_all.py --devices cuda:0,cuda:1 --max_samples 5000 --rcwa_orders 7
```

### 3. 训练数据格式

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

这里 `2` 个通道分别是 `tpp_mag` 和 `tss_mag`。

## 模型部分

### 前向代理模型

`src/model/train_forward.py`

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
- `runs/forward`（如果环境支持 TensorBoard）

### 条件扩散模型

`src/model/train_diffusion.py`

- 用条件图反推结构
- 训练损失由三项加权组成：
  - `loss_diff`（权重 0.645）：v-prediction MSE，主损失
  - `loss_phys`（权重 0.323）：前向代理物理一致性损失，采用 **Straight-Through Estimator（STE）二值化**，前向传给代理模型的是真实 `{0,1}` 二值结构（保证预测准确），反向梯度则直接穿过阈值不断（保证梯度有效传回 UNet）
  - `loss_bin`（权重 0.032）：二值正则，推动预测像素值向 0/1 两极收拢
- 输入结构训练时会从 `[0, 1]` 映射到 `[-1, 1]`
- 训练日志与预览样本也由 `src/model/train_utils.py` 统一管理

当前实现是项目内自定义的 `ConditionalUNet + GaussianDiffusion`，训练入口固定为这一套架构，没有在脚本里提供多架构切换。

运行：

```bash
python src/model/train_diffusion.py
```

输出：

- `checkpoints/diffusion_last.pt`
- `checkpoints/diffusion_best.pt`
- `checkpoints/diffusion_log.csv`
- `checkpoints/diffusion_preview/*.npy`
- `runs/diffusion`（如果环境支持 TensorBoard）

### 基础采样

`src/model/sample.py`

- 输入目标条件 `data/target_cond.npy`
- 生成多个候选结构
- 用前向代理排序，保留误差最小的样本
- 同时保存前向代理预测出的条件图，方便和目标条件直接比较

运行：

```bash
python src/model/sample.py
```

输出目录默认是 `samples/`，主要包括：

- `all_samples.npy`
- `all_pred_cond.npy`
- `target_cond.npy`
- `all_errors.npy`
- `topk_indices.npy`
- `topk_samples.npy`
- `topk_pred_cond.npy`

### 任务驱动推理与拓扑优化

当前推荐的逆设计流程不再依赖旧版 `laplas.py` 的单一二阶目标入口，而是按任务清单统一走：

1. 从训练集里取高分目标样本作为条件输入
2. 用扩散模型生成候选结构
3. 用 RCWA 按各任务原始评分公式重排
4. 取 top-k 候选继续做 RCWA 拓扑优化

核心脚本：

- `src/infer/task_library.py`
  - 维护任务清单与评分逻辑
  - 六类任务的评分公式直接内联写在这里，不再依赖 `data/` 下的打分脚本导入
  - 当前内置任务包括：
    - `p_second_order`
    - `polarization_independent`
    - `polarization_multiplexed`
    - `fourth_order`
    - `lowpass`
    - `st2`
- `src/infer/run_diffusion_tasks.py`
  - 扩散生成 + RCWA 重排
  - 支持 `--eval_mode target_only/full`
  - 会输出目标热力图、top1 热力图、目标波长一维曲线图、top-k 总览图和完整 `summary.json`
- `src/infer/optimize_task_candidates.py`
  - 读取 `run_diffusion_tasks.py` 产出的 case 目录
  - 用 top-k 扩散候选做 RCWA 连续拓扑优化
  - 优化过程中按任务类型切换目标项，并在评估与选优时使用对应任务分数
  - 优化日志会持续打印任务分数、40 度值和各项 loss

示例：

```bash
# 跑一类任务
python src/infer/run_diffusion_tasks.py --tasks p_second_order --num_samples 16 --topk 5 --eval_mode target_only

# 精确跑一个 case
python src/infer/run_diffusion_tasks.py --tasks polarization_multiplexed:1100nm_id11339 --num_samples 16 --topk 5 --eval_mode full

# 对某个 case 的 top-k 候选继续做拓扑优化
python src/infer/optimize_task_candidates.py \
    --case_dir samples/task_infer/<timestamp>/polarization_multiplexed/1100nm_id11339 \
    --max_inits 3
```

`run_diffusion_tasks.py` 的输出目录结构示例：

```text
samples/task_infer/<timestamp>/<task_key>/<case_label>/
├── target_cond_raw.npy
├── task_weight.npy
├── all_samples.npy
├── all_pred_cond_raw.npy
├── task_scores.npy
├── topk_samples.npy
├── topk_pred_cond_raw.npy
├── target_tpp.png
├── target_tss.png
├── target_vs_top1_curves.png
├── top1_structure.png
├── top1_pred_tpp.png
├── top1_pred_tss.png
├── top{k}_overview.png
└── summary.json
```

`summary.json` 里会记录：

- 原始输入样本的任务分数 `original_input_score`
- 扩散候选的 `task_score`
- `weighted_error` / `global_error`
- top-k 排名细节

`optimization_summary.json` 里会记录：

- 原始输入样本分数 `original_input_score`
- 每个 seed 优化后的 `task_score`
- 对应的 `weighted_raw_mae` / `weighted_norm_mae`
- 40 度位置的目标值与优化后值

## 对比实验（Baseline Comparison）

`src/baselines/` 包含四类对比方法：

- `topo_opt`：随机初始化 + 前向代理引导的拓扑优化，最终统一用 RCWA 打分
- `cvae`
- `cgan`
- `diffusion`

当前 `run_eval.py` 已经支持直接复用 `src/infer/task_library.py` 里的任务 case，默认会和 `run_diffusion_tasks.py` 对齐：

- 默认数据集：`data/train_data_20000.npz`
- 默认扩散 checkpoint：`checkpoints/diffusion_best.pt`
- 默认扩散采样：`cfg_scale=1.0`
- 最终 RCWA 评测口径与 `run_diffusion_tasks.py` 统一

### 训练基线模型

```bash
# CVAE（约 200 epochs，有 early stop）
python src/baselines/train/train_cvae.py \
    --data_path data/train_data_20000.npz \
    --save_dir  checkpoints/cvae

# cGAN（200 epochs，n_critic=2）
python src/baselines/train/train_cgan.py \
    --data_path data/train_data_20000.npz \
    --save_dir  checkpoints/cgan
```

两个命令可在两个 tmux 窗口并行运行，互不依赖。

### 运行评估

推荐直接按任务 case 运行，这样目标样本与 task infer 完全一致。例如：

```bash
python src/baselines/eval/run_eval.py \
    --task_case p_second_order:1050nm_id4279
```

只跑单个方法时，可以显式指定 `--methods`：

```bash
python src/baselines/eval/run_eval.py \
    --task_case p_second_order:1050nm_id4279 \
    --methods diffusion
```

如果需要覆盖默认路径，也可以显式传参：

```bash
python src/baselines/eval/run_eval.py \
    --data_path      data/train_data_20000.npz \
    --forward_ckpt   checkpoints/forward_best.pt \
    --stats_path     checkpoints/cond_stats.npz \
    --cvae_ckpt      checkpoints/cvae/cvae_best.pt \
    --cgan_ckpt      checkpoints/cgan/cgan_best.pt \
    --diffusion_ckpt checkpoints/diffusion_best.pt \
    --task_case      p_second_order:1050nm_id4279 \
    --n_samples      32 \
    --save_dir       samples/eval_compare
```

输出：`samples/eval_compare/results.json`，同时打印汇总表格。

> 若某个 checkpoint 不存在，该方法会被自动跳过，不影响其余方法。
>
> 当前 baseline 方法集合默认不包含 `diffusion+guide`。
> 旧的 `--topk_csv + --target_lambda + --target_rank` 路径仍保留兼容，但主线实验更推荐直接使用 `--task_case`。

### 绘制对比图

```bash
python src/baselines/eval/plot_results.py \
    --results  samples/eval_compare/results.json \
    --save_dir samples/eval_compare/figures
```

输出四张图：

| 文件 | 内容 |
|------|------|
| `fig1_main_metrics.png` | 三大指标分组柱状图（best_score / success_rate / spectrum_mae） |
| `fig2_score_boxplot.png` | best_score 箱线图（展示跨 case 稳定性） |
| `fig3_time_vs_quality.png` | 推理时间 vs 质量散点图 |
| `fig4_diversity.png` | 结构多样性对比柱状图 |

### 评估指标说明

| 指标 | 含义 | 越高越好 |
|------|------|----------|
| `best_score` | N 个候选中最优的二阶打分（0~1） | ✓ |
| `top1_success_rate` | score > 0.5 的候选占比 | ✓ |
| `spectrum_mae` | 最优候选预测光谱与目标的 L1 误差 | ✗ |
| `diversity` | N 个候选之间的平均归一化汉明距离 | ✓ |
| `inference_time_s` | 生成 N 个候选的总耗时（秒） | ✗ |



### 光学成像仿真

`src/infer/imag_process/`

基于角谱法（Fourier optics）验证超表面设计的成像效果，包含两个脚本：

#### `image_processing.py` — 成像仿真主脚本

将目标传递函数作用于输入图像（默认中心小方块），输出像平面强度分布。

适配参数：λ = 1000 nm，θ_max = 40° → NA ≈ 0.6428，入射偏振推荐 `x`（p 偏振）。

传递函数来源（精度递增）：

| 模式 | 说明 |
|------|------|
| `--ideal` | 理想 `T=(k_rho/k_max)²`，验证仿真流程 |
| `--sample N` / `--from_infer` | 从 `train_data.npz` 或推理结果读 1D 角度数据，各向同性插值 |
| `--from_kspace` | 从 `scan_kspace.py` 输出的精确 2D 传递函数插值 |

```bash
# 理想二阶传递函数（推荐先跑这个验证流程）
python src/infer/imag_process/image_processing.py --ideal --pol x

# 用训练数据第 0 个样本
python src/infer/imag_process/image_processing.py --sample 0 --pol x

# 用推理结果最优样本
python src/infer/imag_process/image_processing.py --from_infer --pol x

# 用精确 2D 传递函数（需先运行 scan_kspace.py）
python src/infer/imag_process/image_processing.py --from_kspace --pol x
```

输出：`imaging_result.png`（输入图 / 输出强度 / 传递函数截面）

#### `scan_kspace.py` — 2D k 空间传递函数图

利用 C4 + σx 对称性，只需 phi=0°（已有）和 phi=45°（可选新跑）两条扫描线，插值重建完整 k 空间圆盘，生成文献标准图（横纵坐标 kx/k0、ky/k0，颜色代表 |t|，圆外 mask）。

```bash
# 各向同性近似（仅用 phi=0 已有数据，秒出）
python src/infer/imag_process/scan_kspace.py --sample 0

# 精确版（额外跑 phi=45° RCWA，17 次仿真，需 torcwa）
python src/infer/imag_process/scan_kspace.py --sample 0 --run_phi45 --device cuda:0

# 从推理结果加载
python src/infer/imag_process/scan_kspace.py --from_infer
```

输出：`kspace_tpp.png` / `kspace_tss.png`（文献圆形 k 空间图）+ `kspace_result.npz`

---

## 依赖

仓库目前没有完整依赖锁定文件，至少需要：

- Python `>= 3.10`
- `numpy`
- `scipy`
- `torch`
- `torcwa`
- `matplotlib`

说明：

- 批量结构生成默认不依赖 `matplotlib`，只有单独画图时才会导入
- RCWA 部分必须有 `torch + torcwa`
- TensorBoard 可视化依赖 `torch.utils.tensorboard`

## 环境准备

推荐使用 `conda` 单独创建环境。

### 1. 拉取项目

```bash
git clone -b v3 https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

如果你在星宇智算上直接访问 GitHub 较慢，可以给原始 GitHub 链接加一个加速前缀再克隆。常见写法是把：

```text
https://github.com/lsdfd/GraduationProject.git
```

改成：

```text
https://ghfast.top/https://github.com/lsdfd/GraduationProject.git
```

对应命令：

```bash
git clone -b v3 https://ghfast.top/https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

说明：

- 这里的 `ghfast.top/` 只是 GitHub 访问加速前缀，不是仓库地址本身。
- 实际执行时，就是把原来的 GitHub URL 整体接在 `https://ghfast.top/` 后面。
- 如果你已经在服务器上有仓库目录，就不需要重新 clone，直接在仓库里执行 `git checkout v3 && git pull origin v3`。

### 2. 创建 conda 环境

```bash
conda create -n metagen python=3.10 -y
conda init bash（服务器）
source ~/.bashrc（服务器）
conda activate metagen
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果你的 `torch` / `torcwa` 需要匹配特定 CUDA 版本，建议先按官方方式安装对应版本的 `torch`，再安装其余依赖。

### 4. 可选：查看训练曲线

```bash
tensorboard --logdir runs
```

## 当前已知问题

这份仓库仍然是研究代码，不是完全整理好的工程，目前已知缺口有：

1. `pyproject.toml` 只有最小项目信息，没有声明实际运行依赖。
2. 各脚本默认使用仓库根目录下的相对路径 `data/...`、`checkpoints/...`、`samples/...`，建议始终在仓库根目录执行命令。
3. RCWA 批量仿真计算量较大，目前只有日志、定期保存和失败记录，没有断点续跑。
4. `train_data.npz` 中失败点会用 `NaN` 保存，训练前需要自行确认是否要过滤失败样本。
5. 当前模型仍然是研究型轻量实现，不是基于 `diffusers` 或 `accelerate` 的完整工程化扩散框架。

## 建议运行顺序

```bash
python src/dataset/structure/dataset_pre.py --num_samples 5000
python src/dataset/rcwa/rcwa_all.py   ->耗时随结构数、RCWA 阶数和 GPU 数量变化
python src/model/train_forward.py     ->几分钟
python src/model/train_diffusion.py   ->耗时随数据量和训练配置变化
python src/infer/run_diffusion_tasks.py --tasks p_second_order
python src/infer/optimize_task_candidates.py --case_dir samples/task_infer/<timestamp>/<task_key>/<case_label>
```

常用可视化命令：

```bash
# 只想快速查看结构分布（建议先限制数量）
python data/visualize_dataset.py --num_structures 200 --num_tpp 0

# 已经有 train_data.npz 时，同时查看 structures / tpp / tss
python data/visualize_dataset.py --num_structures 200 --num_tpp 200

# 默认会随机/排序采样 1000 个结构与 1000 个频谱做可视化
python data/visualize_dataset.py

# 如果想导出全部分页图，也可以显式取消限制
python data/visualize_dataset.py --num_structures 0 --num_tpp 0
```

说明：

- `--num_structures <= 0` 表示查看全部结构。
- `--num_tpp <= 0` 表示查看全部 `tpp/tss` 响应图。
- 如果当前还没有 `train_data.npz`，就不要运行包含 `tpp/tss` 可视化的命令。

常用二阶打分命令：

```bash
# 对 train_data.npz 里的 tpp_mag 做逐波长二阶打分
python data/score_second_order.py

# 指定输入文件、输出目录和每个波长保留的 top-k
python data/score_second_order.py --in_npz data/train_data.npz --field tpp_mag --out_dir data/second_order_scores --topk 20 --plot_topk 5

# 如果想对 tss_mag 也做同样的打分
python data/score_second_order.py --field tss_mag --out_dir data/second_order_scores_tss
```

说明：

- `data/score_second_order.py` 默认对输入 `npz` 中的全部样本打分，没有写死 `1000` 或其他样本数上限。
- 样本数越多，运行时间越长，输出的 `csv/json/png` 也会更大。
- 主要输出包括 `*_scores.npz`、`*_scores.csv`、`*_summary.json`、`*_top{k}_per_lambda.csv` 以及对应的 top-k 图。

如果环境支持 TensorBoard，可用下面命令查看训练曲线：

```bash
tensorboard --logdir runs
```

## 后续工作方向

### 数据

- 新算法尝试 + 聚类分析，改善结构分布覆盖
- 扩充数据规模到 2w 样本
- 重新定义 score（当前二阶打分函数可优化）
- 扫描到 60° 入射角后重新训练，对标 SOTA

### 模型

- 尝试单波长条件输入（简化条件空间）
- CVAE / cGAN / 拓扑优化 + 各项指标完整对比

### 推理与应用

- 尝试多个目标波长同时优化
- 成像仿真定量指标分析（对比理想 Laplacian 的 PSNR / SSIM）
- 偏振复用、高阶微分、时空微分等扩展场景
