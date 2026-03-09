from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_old_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = np.load(path)
    if not hasattr(data, "files"):
        raise ValueError(f"Input is not an npz file: {path}")

    if "structures" not in data.files:
        raise ValueError("Missing required key: structures")

    structures = np.asarray(data["structures"], dtype=np.uint8)
    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures must be [N,64,64], got {structures.shape}")

    if "tpp_mag" in data.files:
        tpp_mag = np.asarray(data["tpp_mag"], dtype=np.float32)
    elif "tpp_real" in data.files and "tpp_imag" in data.files:
        tpp_real = np.asarray(data["tpp_real"], dtype=np.float32)
        tpp_imag = np.asarray(data["tpp_imag"], dtype=np.float32)
        tpp_mag = np.sqrt(tpp_real**2 + tpp_imag**2).astype(np.float32)
    else:
        raise ValueError("Need tpp_mag or (tpp_real and tpp_imag) in input npz")

    if tpp_mag.ndim != 3:
        raise ValueError(f"tpp_mag must be [N,H,W], got {tpp_mag.shape}")
    if tpp_mag.shape[0] != structures.shape[0]:
        raise ValueError("structures and tpp sample count mismatch")

    if "lambdas" in data.files:
        lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    else:
        lambdas = np.arange(1000.0, 1500.1, 50.0, dtype=np.float32)

    if "thetas" in data.files:
        thetas = np.asarray(data["thetas"], dtype=np.float32)
    else:
        thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)

    return {
        "structures": structures,
        "tpp_mag": tpp_mag,
        "lambdas": lambdas,
        "thetas": thetas,
    }


def build_tss_placeholder(shape: tuple[int, int, int], fill: str) -> np.ndarray:
    if fill == "zero":
        return np.zeros(shape, dtype=np.float32)
    return np.full(shape, np.nan, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert old RCWA npz (real/imag) to new mag-only format"
    )
    parser.add_argument(
        "--in_npz",
        type=Path,
        default=Path("data/mock_train_data.npz"),
        help="Input old-format npz file",
    )
    parser.add_argument(
        "--out_npz",
        type=Path,
        default=Path("data/train_data_mag.npz"),
        help="Output new-format npz file",
    )
    parser.add_argument(
        "--tss_fill",
        choices=["nan", "zero"],
        default="nan",
        help="Placeholder value for tss_mag",
    )
    args = parser.parse_args()

    loaded = load_old_npz(args.in_npz)
    tss_mag = build_tss_placeholder(loaded["tpp_mag"].shape, args.tss_fill)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        structures=loaded["structures"],
        tpp_mag=loaded["tpp_mag"],
        tss_mag=tss_mag,
        lambdas=loaded["lambdas"],
        thetas=loaded["thetas"],
    )

    print(f"input:  {args.in_npz}")
    print(f"output: {args.out_npz}")
    print(f"structures: {loaded['structures'].shape}")
    print(f"tpp_mag:   {loaded['tpp_mag'].shape}")
    print(f"tss_mag:   {tss_mag.shape} (fill={args.tss_fill})")


if __name__ == "__main__":
    main()

