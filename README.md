# Metagen

面向自由形态超表面逆向设计的单波长版本研究原型。

当前 `onelambda` 主线已经直接替换旧的整谱/多波段流程，整个项目只围绕：

- 目标波长：`1000 nm`
- 角度范围：`-40° ~ 40°`
- 条件表示：`[tpp(17), tss(17)]`

也就是：

`自由形态结构 -> RCWA 单波长扫角 -> 前向代理(结构->17点角响应) -> 条件扩散(17点角响应->结构) -> RCWA复核 -> 拓扑优化`

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
- `tss_mag`: `[N, 17]`
- `target_lambda`: `1000.0`
- `thetas`: `[17]`

其中 `thetas = [-40, -35, ..., 40]`。

训练时条件张量统一为：

- `cond`: `[2, 17]`

两个通道分别对应 `tpp` 和 `tss`。

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
- 输出：`[2,17]`

### 4. 训练条件扩散

```bash
python src/model/train_diffusion.py
```

扩散模型学习：

- 条件：`[2,17]`
- 输出：二值结构

训练损失由三部分构成：

- `loss_diff`: v-prediction 去噪损失
- `loss_phys`: 前向代理物理一致性损失，只在 `1000nm` 这一条角度响应上计算
- `loss_bin`: 二值化正则

### 5. 推理

```bash
python src/infer/laplas.py
```

推理流程：

- 从数据集中读取 `1000nm` 下 top-1 模板角响应作为目标
- 扩散模型采样多个候选结构
- 可选物理引导：每步通过前向代理对 `1000nm` 目标角响应做梯度修正
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
- 训练和推理现在都要求单波长数据格式：`structures [N,64,64]`，`tpp_mag/tss_mag [N,17]`。
- 根目录历史 checkpoint、`runs/`、`samples/` 不再作为默认输入；需要先重新训练，或显式传入新的 run 目录产物。

## 最小运行顺序

```bash
python data/extract_onelambda_dataset.py
python data/score_second_order.py --in_npz data/train_data.npz --field tpp_mag --out_dir data/second_order_scores --topk 20 --plot_topk 5
python src/model/train_forward.py
python src/model/train_diffusion.py
python src/infer/laplas.py
python src/infer/optimization.py
```
