import torch
import torcwa
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.materials.materia import Material


def _threshold(x, th=0.5):
    return (x >= th).to(x.dtype)


def _binary_projection(x, beta=10.0):
    # Smooth projection around 0.5 for differentiable optimization.
    return torch.sigmoid(beta * (x - 0.5))


def torcwa_simulation(phy_kwargs, layer, rcwa_orders=7, validity_guard=False, project=True, device=None):
    """
    使用 TORCWA 对单层周期超表面进行 RCWA 电磁仿真，只输出 0 阶透射 Jones t 矩阵（s/p 偏振基）。

    参数:
    1. `phy_kwargs` (dict):
       - `periodicity`: 元胞周期，单位纳米（nm）
       - `h`: 结构层厚度（高度），单位纳米（nm）
       - `lam`: 入射波长，单位纳米（nm）
       - `tet`: 入射角（theta），单位度（degree）
       - `substrate`: 衬底材料名（如 `'SiO2'`）
       - `structure`: 结构材料名（如 `'Si'`）
    2. `layer` (Tensor): 二维结构图（0/1 或 [0,1]）。
    3. `rcwa_orders` (int): RCWA 截断阶数。
    4. `validity_guard` (bool): 保留接口参数，当前未使用。
    5. `project` (bool): 是否先把输入结构投影到二值附近。
    6. `device` (str | torch.device | None):
       - `None`: 若可用则自动使用 `cuda:0`，否则使用 `layer` 当前设备。
       - 其他: 显式指定如 `"cpu"`, `"cuda:0"`。

    返回:
    `dict`:
    - `t_matrix`: 0 阶透射 Jones 矩阵（2x2, complex），基为 [s, p]。
      [[t_ss, t_sp],
       [t_ps, t_pp]]
    - `all`: `t_matrix` 展平后的长度 4 复数向量。

    注:
    - 在 `azi_ang=0` 下，TE 对应 s 偏振，TM 对应 p 偏振。
    """

    _ = validity_guard  # keep API compatibility

    if device is None:
        sim_device = torch.device("cuda:0") if torch.cuda.is_available() else layer.device
    else:
        sim_device = torch.device(device)
    layer = layer.to(sim_device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torcwa.rcwa_geo.Lx = phy_kwargs["periodicity"]  # nm
    torcwa.rcwa_geo.Ly = phy_kwargs["periodicity"]  # nm
    torcwa.rcwa_geo.nx = layer.shape[0]
    torcwa.rcwa_geo.ny = layer.shape[1]
    torcwa.rcwa_geo.grid()
    torcwa.rcwa_geo.edge_sharpness = 10000000.0

    order = [rcwa_orders, rcwa_orders]
    L = [torcwa.rcwa_geo.Lx, torcwa.rcwa_geo.Ly]
    torch.set_num_threads(1)
    sim = torcwa.rcwa(
        freq=1 / phy_kwargs["lam"],  # lam in nm -> freq in 1/nm
        order=order,
        L=L,
        dtype=torch.complex64,
        device=sim_device,
        stable_eig_grad=False,
    )

    # Material tables use um; convert nm -> um.
    lam_um = torch.tensor(float(phy_kwargs["lam"]) / 1000.0, device=sim_device, dtype=torch.float32)
    substrate_eps = Material.apply(phy_kwargs["substrate"], lam_um) ** 2
    structure_eps = Material.apply(phy_kwargs["structure"], lam_um) ** 2
    air_eps = 1.0

    if project:
        layer = _binary_projection(layer, 10.0) if layer.requires_grad else _threshold(layer, 0.5)

    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=phy_kwargs["tet"], azi_ang=0)
    sim.add_layer(thickness=phy_kwargs["h"], eps=(structure_eps * layer + air_eps * (1.0 - layer)))

    sim.solve_global_smatrix()

    zero_order = torch.tensor([[0, 0]], device=sim_device)
    tss = sim.S_parameters(orders=zero_order, direction="f", port="t", polarization="ss").reshape(())
    tsp = sim.S_parameters(orders=zero_order, direction="f", port="t", polarization="sp").reshape(())
    tps = sim.S_parameters(orders=zero_order, direction="f", port="t", polarization="ps").reshape(())
    tpp = sim.S_parameters(orders=zero_order, direction="f", port="t", polarization="pp").reshape(())

    t_matrix = torch.stack([
        torch.stack([tss, tsp]),
        torch.stack([tps, tpp]),
    ])

    return {
        "t_matrix": t_matrix,
        "all": t_matrix.reshape(-1),
    }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layer = torch.zeros((64, 64), device=device)
    layer[16:48, 16:48] = 1.0
    phy_kwargs = {
        "periodicity": 500.0,
        "h": 500.0,
        "lam": 1000.0,
        "tet": 0.0,
        "substrate": "SiO2",
        "structure": "Si",
    }
    t0 = time.perf_counter()
    out = torcwa_simulation(phy_kwargs, layer, rcwa_orders=7)
    print(f"sim_time: {time.perf_counter() - t0:.4f}s")
    print(out["t_matrix"])
