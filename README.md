# MetaEdge AI (毕设原型)

面向课题《人工智能辅助的超材料边缘计算研究》的可运行代码骨架。

当前版本目标：

- 用最小原型跑通流程：`自由形态结构生成 -> 光谱仿真(占位) -> 光谱描述符 -> 逆向设计(检索基线)`
- 为后续接入 `MATLAB + COMSOL` 和 `CGAN/CVAE` 预留清晰接口

## 目录结构

- `src/metaedge_ai/structures.py`：自由形态结构参数与生成
- `src/metaedge_ai/simulator.py`：仿真接口（当前是 mock，后续替换 COMSOL）
- `src/metaedge_ai/descriptors.py`：光谱描述符与综合评分
- `src/metaedge_ai/dataset.py`：数据集生成
- `src/metaedge_ai/inverse.py`：逆向设计基线（最近邻检索）
- `src/metaedge_ai/pipeline.py`：端到端流程
- `src/metaedge_ai/cli.py`：命令行入口
- `run_demo.py`：快速演示脚本

## 快速运行

```bash
cd metaedge_ai
python3 run_demo.py
```

或使用 CLI：

```bash
cd metaedge_ai
python3 -m src.metaedge_ai.cli demo --samples 120
```

## 后续替换路线（建议）

1. `simulator.py`
   - 用 `MatlabEngine` / MATLAB 导出的 `.m` 调用 COMSOL 接口替换 `MockSpectrumSimulator`
2. `descriptors.py`
   - 加入论文里的光谱综合评价指标（多目标加权、带宽/峰值/角度稳定性）
3. `inverse.py`
   - 保留 `fit/predict` 接口，新增 `CVAEInverseDesigner`、`CGANInverseDesigner`
4. `dataset.py`
   - 支持真实仿真数据缓存、版本管理、训练/验证切分

