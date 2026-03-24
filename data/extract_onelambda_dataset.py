from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 1000nm one-lambda dataset from legacy full-spectrum npz.")
    parser.add_argument("--src", default=str(ROOT / "train_data_20000.npz"))
    parser.add_argument("--out", default=str(ROOT / "train_data.npz"))
    parser.add_argument("--target_lambda", type=float, default=1000.0)
    args = parser.parse_args()

    src = resolve(args.src)
    out = resolve(args.out)

    data = np.load(src)
    required = {"structures", "tpp_mag", "tss_mag", "lambdas", "thetas"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing fields in {src}: {sorted(missing)}")

    structures = np.asarray(data["structures"], dtype=np.uint8)
    tpp = np.asarray(data["tpp_mag"], dtype=np.float32)
    tss = np.asarray(data["tss_mag"], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)

    lam_idx = int(np.argmin(np.abs(lambdas - float(args.target_lambda))))
    picked_lambda = float(lambdas[lam_idx])

    tpp_row = tpp[:, lam_idx, :]
    tss_row = tss[:, lam_idx, :]
    valid = np.isfinite(tpp_row).all(axis=1) & np.isfinite(tss_row).all(axis=1)

    structures = structures[valid]
    tpp_row = tpp_row[valid]
    tss_row = tss_row[valid]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        structures=structures,
        tpp_mag=tpp_row.astype(np.float32),
        tss_mag=tss_row.astype(np.float32),
        target_lambda=np.asarray(picked_lambda, dtype=np.float32),
        thetas=thetas.astype(np.float32),
    )

    print(f"source={src}")
    print(f"target_lambda={picked_lambda:.1f} nm (requested {args.target_lambda:.1f} nm)")
    print(f"saved={out}")
    print(f"structures={structures.shape}")
    print(f"tpp_mag={tpp_row.shape}")
    print(f"tss_mag={tss_row.shape}")


if __name__ == "__main__":
    main()
