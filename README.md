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

- 默认生成 `1000` 个 `64x64` 二值结构
- 每个样本使用不同随机种子
- 结构满足 `C4 + sigma_x` 对称
- 约定 `1 = 材料`，`0 = 空气`
- 输出到 `data/structures/structures.npy`

运行：

```bash
python src/dataset/structure/dataset_pre.py
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

- `lambda = 1000, 1050, ..., 1500`，共 `11` 个点
- `theta = -40, -35, ..., 40`，共 `17` 个点

默认取前 `1000` 个结构（可用 `--max_samples` 调整），默认 `rcwa_orders=7`。输出到 `data/train_data.npz`，并额外写：

- `data/rcwa.log`
- `data/train_data_failures.json`

运行：

```bash
python src/dataset/rcwa/rcwa_all.py
```

如果要修改样本数或阶数：

```bash
python src/dataset/rcwa/rcwa_all.py --max_samples 300 --rcwa_orders 7
```

如果要多卡并行（按结构分片）：

```bash
python src/dataset/rcwa/rcwa_all.py --devices cuda:0,cuda:1 --max_samples 1000 --rcwa_orders 7
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
git clone https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

### 2. 创建 conda 环境

```bash
conda create -n metagen python=3.10 -y
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
python src/dataset/structure/dataset_pre.py   ->几秒钟
python src/dataset/rcwa/rcwa_all.py   ->1000个数据约4.5~5小时（单卡）
python src/model/train_forward.py     ->几分钟
python src/model/train_diffusion.py   ->1000个数据大概耗时60min
python src/model/sample.py
```

如果环境支持 TensorBoard，可用下面命令查看训练曲线：

```bash
tensorboard --logdir runs
```
