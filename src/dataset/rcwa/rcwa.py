import math
import time
from pathlib import Path
import sys

import torch
import torcwa

# Keep backward compatibility with the original project layout.
PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else None
if PROJECT_ROOT is not None:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataset.materials.materia import Material  # type: ignore
except Exception:
    Material = None


def _threshold(x, th=0.5):
    return (x >= th).to(x.dtype)


def _binary_projection(x, beta=10.0):
    # Smooth projection around 0.5 for differentiable optimization.
    return torch.sigmoid(beta * (x - 0.5))


def _to_complex_tensor(value, device, dtype=torch.complex64):
    return torch.as_tensor(value, device=device, dtype=dtype)


def build_hollow_brick_mask(periodicity, L_outer, W_inner, nx=256, ny=None, device=None, dtype=torch.float32):
    """Build a centered square hollow-brick mask in one unit cell."""
    if ny is None:
        ny = nx
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    x = torch.linspace(-periodicity / 2.0, periodicity / 2.0, steps=nx, device=device, dtype=dtype)
    y = torch.linspace(-periodicity / 2.0, periodicity / 2.0, steps=ny, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    outer = (xx.abs() <= L_outer / 2.0) & (yy.abs() <= L_outer / 2.0)
    inner = (xx.abs() <= W_inner / 2.0) & (yy.abs() <= W_inner / 2.0)
    return (outer & (~inner)).to(dtype)


def _resolve_eps(*, material_name=None, lam_um=None, device=None, n_override=None, eps_override=None):
    """
    Resolve permittivity in a generic way.

    Priority:
    1) eps_override
    2) n_override
    3) material_name == air / vacuum
    4) project Material database
    """
    if eps_override is not None:
        return _to_complex_tensor(eps_override, device=device)

    if n_override is not None:
        n = _to_complex_tensor(n_override, device=device)
        return n * n

    if material_name is None:
        raise ValueError("Material is not specified. Provide material_name, n_override, or eps_override.")

    name = str(material_name).strip().lower()
    if name in {"air", "vacuum"}:
        return _to_complex_tensor(1.0, device=device)

    if Material is None:
        raise ImportError(
            "Material database is unavailable. Please either make dataset.materials.materia.Material importable, "
            "or pass n_input/n_output/n_structure (or eps_input/eps_output/eps_structure) in phy_kwargs."
        )

    if lam_um is None:
        raise ValueError("lam_um is required when using Material database lookup.")

    n_val = Material.apply(material_name, lam_um)
    n_val = _to_complex_tensor(n_val, device=device)
    return n_val * n_val


# Backward-compatible alias name kept unchanged.
def torcwa_simulation(phy_kwargs, layer, rcwa_orders=13, validity_guard=False, project=True, device=None):
    """
    使用 TORCWA 对单层周期结构进行 RCWA 电磁仿真，只输出 0 阶透射 Jones t 矩阵（s/p 偏振基）。

    兼容旧接口，同时补上三个普适修正：
    1) 角度单位显式处理：默认按 degree 输入，内部自动转为 radian。
    2) 输入/输出介质分开设置：避免把 substrate 默认当作入射侧。
    3) 截断阶数默认从 7 调整到 9：更适合作为通用默认值；批量粗扫可降到 7，正式结果建议做 9/11/13 收敛检查。

    主要参数（phy_kwargs）:
    - periodicity: 元胞周期，单位 nm
    - h: 结构层厚度，单位 nm
    - lam: 波长，单位 nm
    - tet: 入射极角 theta
    - phi: 方位角，默认 0
    - angle_unit: 'deg' 或 'rad'，默认 'deg'
    - structure: 结构材料名
    - substrate: 向后兼容字段；若未显式提供 output_medium，则把 substrate 视为输出侧介质
    - input_medium: 输入侧介质名，默认 'air'
    - output_medium: 输出侧介质名，默认 substrate，否则 'air'
    - angle_layer: 角度参考层，默认 'input'

    也支持直接给常数光学参数覆盖材料库：
    - n_input / n_output / n_structure
    - eps_input / eps_output / eps_structure

    返回:
    - t_matrix: 2x2 复数 Jones 透射矩阵
    - all: 展平后的长度 4 复数向量
    - tpp / tss: 复数透射系数
    - tpp_mag / tss_mag: 对应模值
    """
    _ = validity_guard  # keep API compatibility

    if device is None:
        sim_device = torch.device("cuda:0") if torch.cuda.is_available() else layer.device
    else:
        sim_device = torch.device(device)
    layer = layer.to(sim_device)

    if sim_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    torcwa.rcwa_geo.Lx = float(phy_kwargs["periodicity"])
    torcwa.rcwa_geo.Ly = float(phy_kwargs.get("periodicity_y", phy_kwargs["periodicity"]))
    torcwa.rcwa_geo.nx = int(layer.shape[0])
    torcwa.rcwa_geo.ny = int(layer.shape[1])
    torcwa.rcwa_geo.grid()
    torcwa.rcwa_geo.edge_sharpness = float(phy_kwargs.get("edge_sharpness", 10000000.0))

    order = [int(rcwa_orders), int(rcwa_orders)]
    L = [torcwa.rcwa_geo.Lx, torcwa.rcwa_geo.Ly]

    if "num_threads" in phy_kwargs:
        torch.set_num_threads(int(phy_kwargs["num_threads"]))
    else:
        torch.set_num_threads(1)

    sim = torcwa.rcwa(
        freq=1.0 / float(phy_kwargs["lam"]),  # lam in nm -> freq in 1/nm
        order=order,
        L=L,
        dtype=torch.complex64,
        device=sim_device,
        stable_eig_grad=False,
    )

    lam_um = torch.tensor(float(phy_kwargs["lam"]) / 1000.0, device=sim_device, dtype=torch.float32)

    input_medium = phy_kwargs.get("input_medium", "air")
    output_medium = phy_kwargs.get("output_medium", phy_kwargs.get("substrate", "air"))
    structure_medium = phy_kwargs.get("structure")

    input_eps = _resolve_eps(
        material_name=input_medium,
        lam_um=lam_um,
        device=sim_device,
        n_override=phy_kwargs.get("n_input"),
        eps_override=phy_kwargs.get("eps_input"),
    )
    output_eps = _resolve_eps(
        material_name=output_medium,
        lam_um=lam_um,
        device=sim_device,
        n_override=phy_kwargs.get("n_output"),
        eps_override=phy_kwargs.get("eps_output"),
    )
    structure_eps = _resolve_eps(
        material_name=structure_medium,
        lam_um=lam_um,
        device=sim_device,
        n_override=phy_kwargs.get("n_structure"),
        eps_override=phy_kwargs.get("eps_structure"),
    )
    background_eps = _resolve_eps(
        material_name=phy_kwargs.get("background_medium", "air"),
        lam_um=lam_um,
        device=sim_device,
        n_override=phy_kwargs.get("n_background"),
        eps_override=phy_kwargs.get("eps_background"),
    )

    if project:
        layer = _binary_projection(layer, float(phy_kwargs.get("projection_beta", 10.0))) if layer.requires_grad else _threshold(layer, float(phy_kwargs.get("threshold", 0.5)))

    angle_unit = str(phy_kwargs.get("angle_unit", "deg")).lower()
    tet = float(phy_kwargs.get("tet", 0.0))
    phi = float(phy_kwargs.get("phi", 0.0))
    if angle_unit in {"deg", "degree", "degrees"}:
        tet = math.radians(tet)
        phi = math.radians(phi)
    elif angle_unit not in {"rad", "radian", "radians"}:
        raise ValueError(f"Unsupported angle_unit: {angle_unit}")

    sim.add_input_layer(eps=input_eps)
    sim.add_output_layer(eps=output_eps)
    sim.set_incident_angle(
        inc_ang=tet,
        azi_ang=phi,
        angle_layer=str(phy_kwargs.get("angle_layer", "input")),
    )
    sim.add_layer(
        thickness=float(phy_kwargs["h"]),
        eps=(structure_eps * layer + background_eps * (1.0 - layer)),
    )

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
        "tss": tss,
        "tpp": tpp,
        "tss_mag": torch.abs(tss),
        "tpp_mag": torch.abs(tpp),
    }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layer = torch.zeros((64, 64), device=device)
    layer[16:48, 16:48] = 1.0

    phy_kwargs = {
        "periodicity": 500.0,
        "h": 500.0,
        "lam": 1000.0,
        "tet": 10.0,
        "phi": 0.0,
        "angle_unit": "deg",
        "input_medium": "air",
        "output_medium": "glass",
        "structure": "Si",
        # Self-contained smoke test: use constant indices so this file can run
        # even when the external material database is unavailable.
        "n_input": 1.0,
        "n_output": 1.5,
        "n_structure": 3.4,
    }

    t0 = time.perf_counter()
    out = torcwa_simulation(phy_kwargs, layer, rcwa_orders=5, project=False, device=device)
    dt = time.perf_counter() - t0
    print(f"device: {device}")
    print(f"sim_time: {dt:.4f}s")
    print(out["t_matrix"])
    print(f"|tpp| = {torch.abs(out['tpp']).item():.6f}")
