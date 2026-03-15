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
└── model/
    ├── dataset.py      # 训练数据读取
    ├── models.py       # 前向代理 + 条件 UNet
    ├── diffusion.py    # 扩散过程
    ├── train_utils.py  # 训练日志与可视化工具
    ├── train_forward.py
    ├── train_diffusion.py
    └── sample.py
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
- 同时使用前向代理提供的 physics loss
- 输入结构训练时会从 `[0, 1]` 映射到 `[-1, 1]`
- 当前 U-Net 比最初版本更深，并在低分辨率层加入轻量 attention
- 训练日志与预览样本也由 `src/model/train_utils.py` 统一管理

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

### 采样

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

## Infer 部分

### 二阶目标扩散推理

`src/infer/laplas.py`

- 当前默认在 `1000 nm` 构造二阶目标，目标曲线与 `|sin(theta)|^2` 成正比
- 非目标波长会沿波长方向平滑过渡到背景/模板谱
- 当前默认只扫 1 组目标参数：
  - `floor_0p00_off_0p90`
- 对该目标生成多个候选结构
- 当前会直接调用 RCWA 复核全部候选，再按二阶分数排序保存 top 结果

运行：

```bash
python src/infer/laplas.py
```

常用参数示例：

```bash
python src/infer/laplas.py --num_samples 64 --cfg_scale 3.5 --topk_second 8 --target_lambda 1000
```

输出目录默认在 `samples/laplas/<timestamp>/<case_name>/`，每个 case 主要包括：

- `target_cond_raw.npy`
- `all_samples.npy`
- `all_pred_cond_raw.npy`
- `all_errors.npy`
- `topk_samples.npy`
- `topk_second_samples.npy`
- `topk_second_pred_cond_raw.npy`
- `second_order_metrics.json`
- `summary.json`
- `target_tpp.png`
- `best_tpp.png`
- `top*_second_order.png`

### 拓扑优化

`src/infer/optimization.py`

- 读取 `laplas.py` 生成的目标条件和初始结构
- 默认优先使用 `topk_second_samples.npy` 作为初始化
- 直接基于 RCWA 做多起点拓扑优化
- 当前优化目标以目标波长处的二阶角响应为主，同时加入外侧单调性、二值化和 TV 正则
- 最终按二阶分数排序输出

运行：

```bash
python src/infer/optimization.py
```

常用参数示例：

```bash
python src/infer/optimization.py --steps 500 --lr 0.02 --max_inits 5 --target_lambda 1000
```

输出目录默认在 `samples/optimized/<timestamp>/candidate_XX/`，主要包括：

- `optimized_continuous.npy`
- `optimized_binary.npy`
- `optimized_rcwa_tpp_row.npy`
- `optimized_rcwa_tss_row.npy`
- `optimized_rcwa_continuous_tpp_row.npy`
- `optimized_rcwa_continuous_tss_row.npy`
- `optimized_continuous.png`
- `optimized_binary.png`
- `optimized_second_order_curve.png`
- `optimization_log.json`

根目录还会汇总：

- `optimization_summary.json`

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
python src/model/sample.py
python src/infer/laplas.py
python src/infer/optimization.py
```

常用可视化命令：

```bash
# 只想快速查看结构分布（建议先限制数量）
python data/visualize_dataset.py --num_structures 200 --num_tpp 0

# 已经有 train_data.npz 时，同时查看 structures / tpp / tss
python data/visualize_dataset.py --num_structures 200 --num_tpp 200

# 如果想导出全部分页图，也可以直接不限制数量
python data/visualize_dataset.py
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
