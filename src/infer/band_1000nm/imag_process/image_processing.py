# -*- coding: utf-8 -*-
"""
1000 nm 波段傅里叶光学仿真入口。

Usage
-----
  python src/infer/band_1000nm/imag_process/image_processing.py
  python src/infer/band_1000nm/imag_process/image_processing.py --from_infer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from infer.imag_process.image_processing import (  # noqa: E402
    POL_MAP, NA, THETA_MAX, LAMBDAS, NX, NY,
    make_ideal_transfer,
    load_from_training, load_from_infer, load_from_kspace,
    build_2d_transfer,
    run_fourier_optics,
    plot_results,
)

_DEFAULT_LAMBDA_NM = 1000.0
_IMAG_PROCESS_DIR = _HERE


def main():
    parser = argparse.ArgumentParser(description="Fourier optics imaging simulation @ 1000 nm")
    parser.add_argument("--pol",         default="x", choices=list(POL_MAP.keys()))
    parser.add_argument("--lambda_nm",   type=float, default=_DEFAULT_LAMBDA_NM)
    parser.add_argument("--ideal",       action="store_true")
    parser.add_argument("--sample",      type=int, default=0)
    parser.add_argument("--from_infer",  action="store_true")
    parser.add_argument("--infer_dir",   default=None)
    parser.add_argument("--from_kspace", action="store_true")
    parser.add_argument("--kspace_npz",  default=None)
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    lambda_nm = args.lambda_nm
    K0    = 2 * np.pi / (lambda_nm * 1e-9)
    K_MAX = K0 * NA
    lam_idx = int(np.argmin(np.abs(LAMBDAS - lambda_nm)))
    print(f"λ = {lambda_nm:.0f} nm  →  LAMBDAS[{lam_idx}] = {LAMBDAS[lam_idx]:.0f} nm")

    kx_arr = np.linspace(-K0, K0, NX)
    ky_arr = np.linspace(-K0, K0, NY)
    KX, KY = np.meshgrid(kx_arr, ky_arr)

    if args.ideal:
        T_ss, T_pp = make_ideal_transfer(KX, KY, K_MAX)
        src_label  = "ideal  T = (k_rho / k_max)²"
    elif args.from_kspace:
        npz_path = (Path(args.kspace_npz) if args.kspace_npz
                    else _IMAG_PROCESS_DIR / "kspace_result.npz")
        if not npz_path.exists():
            print(f"找不到 {npz_path}，请先运行 scan_kspace.py", file=sys.stderr); sys.exit(1)
        T_ss, T_pp = load_from_kspace(npz_path, KX, KY, K0, K_MAX)
        src_label  = f"kspace: {npz_path.name}"
    elif args.from_infer:
        if args.infer_dir is None:
            for d in [_ROOT / "samples" / "laplas_1000nm", _ROOT / "samples" / "laplas"]:
                runs = sorted(d.iterdir()) if d.exists() else []
                if runs:
                    args.infer_dir = runs[-1]; break
        if args.infer_dir is None:
            print("找不到推理结果，请用 --sample", file=sys.stderr); sys.exit(1)
        print(f"[auto] infer_dir = {args.infer_dir}")
        t_ss_1d, t_pp_1d = load_from_infer(Path(args.infer_dir), lam_idx)
        T_ss = build_2d_transfer(t_ss_1d, KX, KY, K0, K_MAX)
        T_pp = build_2d_transfer(t_pp_1d, KX, KY, K0, K_MAX)
        src_label = f"infer: {Path(args.infer_dir).name}"
    else:
        t_ss_1d, t_pp_1d = load_from_training(args.sample, lam_idx)
        T_ss = build_2d_transfer(t_ss_1d, KX, KY, K0, K_MAX)
        T_pp = build_2d_transfer(t_pp_1d, KX, KY, K0, K_MAX)
        src_label = f"train sample {args.sample}"

    print(f"Source : {src_label}")
    e_in = POL_MAP[args.pol]
    I_out, I_in = run_fourier_optics(T_ss, T_pp, e_in, KX, KY, K0)
    out_path = Path(args.out) if args.out else _IMAG_PROCESS_DIR / f"imaging_result_{lambda_nm:.0f}nm.png"
    plot_results(I_in, I_out, T_ss, T_pp, KX, K_MAX, lambda_nm, args.pol, out_path)


if __name__ == "__main__":
    main()
