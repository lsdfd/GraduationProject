# MetaDiffusion 最小复现骨架（面向论文方法主线）

本目录是一个“最小可维护版”的 MetaDiffusion 复现骨架，目标是先把论文主方法跑通，而不是一次性搭完整实验平台。

当前范围：
- 只实现 `MetaDiffusion` 主模型（扩散模型）
- 不包含基线模型（PNN / SLMGAN / WGAN-GP / cVAE）
- 不包含 CST/代理求解器验证流程
- 当前默认使用“假数据”打通训练与采样链路

适用场景：
- 先确认论文算法逻辑（扩散训练 + CFG采样）是否实现正确
- 后续再替换成真实数据读取器

## 与论文/补充材料对齐的关键点

本实现已经对齐的核心方法细节（来自主文 + 补充材料）：
- 扩散步数 `T=500`
- 线性 `beta` 调度：`1e-4 -> 0.02`
- 训练目标：预测噪声（MSE）
- `classifier-free guidance` 训练：`10%` 条件置零
- 推理时条件/无条件混合：`(1+w)e_cond - w e_uncond`
- 采样引导强度：`w=6.0`
- 输出阈值二值化：`0.5`
- 图案表示：利用对称先验，将 `64x64` 结构简化为左上角 `32x32` 作为生成目标
- 条件向量维度：`55 = 52(S参数) + 3(W1,H2,N2)`

## 当前实现与论文“完全一致”之间的差异

这份代码是“论文方法最小骨架”，不是全文严格一比一复现。主要差异：
- 当前 `data.py` 内置的假数据集用于打通流程，不是论文真实数据集
- 未接入 `PNN` 或 `CST` 做物理前向验证
- 未实现论文对比实验（SLMGAN / WGAN-GP / cVAE）
- `model.py` 是贴近图S1思想的紧凑 U-Net，不是逐层完全拷贝论文网络图

如果后续你要做“论文级完整复现”，优先补：
1. 真实数据读取与字段还原
2. 验证集/测试集 MAE 评估流程
3. PNN 或 CST 前向验证

## 文件结构（最小版）

- `data.py`
  - 假数据生成（符合论文输入形状）
  - 条件向量构造（`52 + 3 = 55`）
  - `64x64 -> 32x32` 象限提取与 `64x64` 对称重建

- `model.py`
  - MetaDiffusion 使用的 U-Net 风格噪声预测网络 `εθ(x_t, t, c)`
  - 时间步嵌入（time embedding）
  - 条件嵌入（condition embedding）
  - 解码器阶段嵌入注入（贴近补充材料图S1）

- `train.py`
  - 训练脚本入口
  - 扩散公式实现（前向加噪、损失、反向采样步骤）
  - CFG 混合公式
  - 训练循环（含 `10%` 条件置零）

- `infer.py`
  - 从训练好的模型采样生成结构
  - 使用 `w=6.0` 做 CFG 推理
  - `sigmoid + 0.5` 阈值二值化
  - 对称恢复完整 `64x64` 结构

## 依赖（Dependencies）

### 必需依赖
- `python >= 3.10`（建议）
- `torch`（PyTorch）

### 可选依赖
- `numpy`（`infer.py` 保存文本矩阵时使用）

### 安装方式（推荐）

```bash
cd "src/model/Difussion Model/metadiffusion_min"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install torch numpy
```

如果你使用的是 Apple Silicon / CUDA 环境，`torch` 安装命令可能需要按你的平台替换为官方推荐命令。

## 部署 / 执行方式（本地运行）

这里的“部署”指本地可运行环境的准备和执行，不涉及服务化部署。

### 1. 训练（假数据）

```bash
cd "src/model/Difussion Model/metadiffusion_min"
python3 train.py --epochs 2 --steps-per-epoch 20 --dataset-len 512 --outdir outputs --device cpu
```

说明：
- `--dataset-len`：假数据集长度
- `--steps-per-epoch`：每个 epoch 实际跑多少个 batch（便于快速调试）
- `--outdir`：保存 checkpoint 的目录
- `--device`：`cpu` 或 `cuda`

训练完成后会得到类似文件：
- `outputs/metadiffusion_epoch002.pt`

### 2. 采样（推理/预测）

```bash
cd "src/model/Difussion Model/metadiffusion_min"
python3 infer.py --ckpt outputs/metadiffusion_epoch002.pt --num-samples 2 --outdir samples --device cpu
```

输出内容（文本矩阵形式，避免图像依赖）：
- `infer_000_quadrant_cont.txt`：连续值 `32x32`
- `infer_000_quadrant_bin.txt`：二值化 `32x32`
- `infer_000_full64_bin.txt`：对称恢复后的 `64x64`

## 输入/输出张量约定（后续接真实数据时最重要）

### 模型训练输入
- `x0`: `Tensor[B, 1, 32, 32]`
  - 目标结构图（左上角象限）
- `cond`: `Tensor[B, 55]`
  - `52维S参数 + 3维几何参数`
- `t`: `Tensor[B]`
  - 扩散时间步（`0 ~ 499`）

### 模型输出
- `pred_noise`: `Tensor[B, 1, 32, 32]`
  - 预测噪声（不是直接输出结构）

### 采样输出
- 连续图 `x0_cont`（通过 `sigmoid` 转为概率）
- 二值图 `x0_bin`（阈值 `0.5`）
- 对称恢复后的 `64x64` 结构

## 如何把真实数据嵌入当前模型（设计说明）

你后续只需要主要改 `data.py` 中的数据读取/预处理部分，尽量不要动 `model.py / train.py` 的扩散核心逻辑。

理想的真实数据字段（与论文对齐）应当能提供：
- `pattern_full`: `64x64`
- `s_real`: `26`
- `s_imag`: `26`
- `geom`: `3`（`W1, H2, N2`）

然后在 `data.py` 的数据集部分转换为：
- `x0 = upper_left_quadrant(pattern_full)` -> `[1,32,32]`
- `cond = concat([s_real, s_imag, geom])` -> `[55]`

## 关于你当前下载的 `dataset/`（重要）

你当前放在 `src/model/Difussion Model/dataset` 的数据文件看起来并不直接符合论文主实验的数据格式，原因包括：
- `pattern.npy` 形状是 `64 x 320 x 97520`（不是常见的单样本 `64x64` 堆叠形式）
- `real_part.npy` / `imaginary_part.npy` 当前表现为一维 `(97520,)`
- 这不符合论文中每个结构对应“多频点 S 参数曲线（52维）”的描述

这意味着：
- 该数据很可能在 `.mat -> .npy` 导出时丢失了维度信息（被扁平化）
- 或者它不是论文 MetaDiffusion 主实验使用的那一版数据结构

在接入真实训练前，建议先完成：
1. 还原 `pattern.npy` 的真实语义（`320` 是否由多块拼接而成）
2. 还原 `real/imag` 的频率维度（不应只有单值标签）

## 常见问题（FAQ）

### 1. 为什么 `infer.py` 里做了 `sigmoid`？
- 当前模型输出的是噪声反推得到的连续值，不是严格的二值图。
- 为了得到可阈值化的结构图，先映射到 `[0,1]` 再做 `0.5` 阈值。

### 2. 为什么不用真实数据直接训练？
- 你当前下载的数据格式和论文描述不一致，直接接入会导致条件维度和样本语义错误。
- 先用假数据把扩散主链路跑通，是为了隔离“模型实现问题”和“数据格式问题”。

### 3. 现在这份代码能算论文里的 MAE 吗？
- 不能。
- 还缺前向求解器（PNN 或 CST）以及验证/测试评估流程。

## 后续建议（按优先级）

1. 先还原真实数据的 shape 与字段语义（尤其 `real/imag` 的频率维度）
2. 将 `data.py` 中的 `FakeMetaAtomDataset` 替换为真实数据读取器
3. 加入验证集评估流程（至少统计训练损失外的结构质量/条件误差）
4. 再决定是否补 PNN/CST 与论文图表复现
