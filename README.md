# Metagen

面向自由形态超表面逆向设计的单波长版本研究原型。

当前 `onelambda` 主线已经直接替换旧的整谱/多波段流程，整个项目只围绕：

- 目标波长：`1000 nm`
- 角度范围：`-40° ~ 40°`
- 条件表示：`tpp(17)`

也就是：

`自由形态结构 -> RCWA 单波长扫角 -> 前向代理(结构->17点 tpp) -> 条件扩散(17点 tpp->结构) -> RCWA复核 -> 拓扑优化`

## 当前目录

```text
src/
├── dataset/
│   ├── materials/
│   ├── rcwa/
│   │   ├── rcwa.py
│   │   └── rcwa_all.py
│   └── structure/
├── infer/
│   ├── common.py
│   ├── laplas.py
│   ├── optimization.py
│   ├── validate_surrogate_top.py
│   └── imag_process/
├── model/
│   ├── dataset.py
│   ├── diffusion.py
│   ├── models.py
│   ├── train_diffusion.py
│   ├── train_forward.py
│   └── train_utils.py
```

## 数据格式

`data/train_data.npz` 当前约定：

- `structures`: `[N, 64, 64]`
- `tpp_mag`: `[N, 17]`
- `target_lambda`: `1000.0`
- `thetas`: `[17]`

其中 `thetas = [-40, -35, ..., 40]`。

训练时条件张量统一为：

- `cond`: `[17]`

## 流程

### 1. 结构生成

```bash
python src/dataset/structure/dataset_pre.py --num_samples 5000
```

输出：

- `data/structures/structures.npy`

### 2. RCWA 单波长扫角

```bash
python src/dataset/rcwa/rcwa_all.py
```

这一步只计算 `1000nm` 下的 `17` 个角度点。

输出：

- `data/train_data.npz`
- `data/train_data_failures.json`
- `data/rcwa.log`

如果已经有旧分支留下的 `data/train_data_20000.npz`，当前主线更推荐直接提取：

```bash
python data/extract_onelambda_dataset.py
```

它会从旧的 `[N,11,17]` 数据里抽出 `1000nm` 这一行，生成新的单波长训练集 `data/train_data.npz`。

### 3. 训练前向代理

```bash
python src/model/train_forward.py
```

前向代理学习：

- 输入：`[1,64,64]`
- 输出：`[17]`

### 4. 训练条件扩散

```bash
python src/model/train_diffusion.py
```

扩散模型学习：

- 条件：`[17]`
- 输出：二值结构

训练损失由三部分构成：

- `loss_diff`: v-prediction 去噪损失
- `loss_phys`: 前向代理物理一致性损失，只在 `1000nm` 的 `tpp(theta)` 上计算
- `loss_bin`: 二值化正则

### 5. 推理

```bash
python src/infer/laplas.py
```

推理流程：

- 直接构造理想二阶微分目标 `tpp(theta) ~ |sin(theta)|^2`
- 扩散模型采样多个候选结构
- 可选物理引导：每步通过前向代理对理想 `tpp(theta)` 做梯度修正
- 用真实 RCWA 单波长扫角复核
- 按二阶响应分数排序保存

### 6. 拓扑优化

```bash
python src/infer/optimization.py
```

拓扑优化直接以 `1000nm` 下的 RCWA 角响应为目标，做多起点优化。

## 说明

- 旧的多波段入口、整谱目标构造、baseline 对比框架已经从 `onelambda` 主线移除。
- 当前仓库默认只支持 `1000nm` 单波长工作流。
- 训练和推理现在都要求单波长数据格式：`structures [N,64,64]`，`tpp_mag [N,17]`。
- 根目录历史 checkpoint、`runs/`、`samples/` 不再作为默认输入；需要先重新训练，或显式传入新的 run 目录产物。

## 环境准备

推荐在服务器上单独创建一个 `conda` 环境。

### 1. 拉取 `onelambda` 分支

如果服务器访问 GitHub 正常：

```bash
git clone -b onelambda --single-branch https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

如果服务器直连 GitHub 较慢，可以使用加速前缀：

```bash
git clone -b onelambda --single-branch https://ghfast.top/https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
```

如果服务器上已经有仓库目录，只需要更新：

```bash
git fetch origin
git checkout onelambda
git pull origin onelambda
```

### 2. 创建环境

```bash
conda create -n metagen python=3.10 -y
conda activate metagen
```

如果你在服务器上第一次用 `conda`，通常还需要先执行：

```bash
conda init bash
source ~/.bashrc
conda activate metagen
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果你的服务器 CUDA 版本和默认 `torch` 轮子不匹配，建议先按官方方式安装对应版本的 `torch`，再安装剩余依赖。

### 4. 检查数据

当前主线默认使用：

- `data/train_data.npz`

如果你只有旧格式的：

- `data/train_data_20000.npz`

就先提取：

```bash
python data/extract_onelambda_dataset.py
```

## 服务器部署流程

下面是一套从零到可推理的最小部署顺序。

### 第一步：准备代码和环境

```bash
git clone -b onelambda --single-branch https://ghfast.top/https://github.com/lsdfd/GraduationProject.git
cd GraduationProject
conda create -n metagen python=3.10 -y
conda activate metagen
pip install -r requirements.txt
```

### 第二步：准备 one-lambda 数据

如果已经有旧的 `data/train_data_20000.npz`：

```bash
python data/extract_onelambda_dataset.py
```

如果没有旧数据，而是从头生成：

```bash
python src/dataset/structure/dataset_pre.py --num_samples 5000
python src/dataset/rcwa/rcwa_all.py
```

### 第三步：训练两个模型

```bash
python src/model/train_forward.py
python src/model/train_diffusion.py
```

训练输出会写到：

- `checkpoints/forward_runs/...`
- `checkpoints/diffusion_runs/...`

并且会自动更新：

- `checkpoints/latest_forward_run.txt`
- `checkpoints/latest_diffusion_run.txt`

### 第四步：推理和优化

```bash
python src/infer/laplas.py
python src/infer/optimization.py
```

`laplas.py` 会自动读取最新的 forward/diffusion run 目录中的 checkpoint 和 `cond_stats.npz`。  
`optimization.py` 会继续读取 `laplas.py` 最新输出的目标和候选结构。

## 最小运行顺序

```bash
python data/extract_onelambda_dataset.py
python data/score_second_order.py --in_npz data/train_data.npz --field tpp_mag --out_dir data/second_order_scores --topk 20 --plot_topk 5
python src/model/train_forward.py
python src/model/train_diffusion.py
python src/infer/laplas.py
python src/infer/optimization.py
```
