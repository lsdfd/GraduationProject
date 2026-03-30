import argparse
import subprocess
import sys
from pathlib import Path

from diffusion_presets import DIFFUSION_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple diffusion presets sequentially for comparison.")
    parser.add_argument(
        "--presets",
        default="baseline_cnn_cross,token_cross_v3,hybrid_cross_v1,hybrid_selfcond_v1",
        help=f"Comma-separated preset names. Available: {', '.join(sorted(DIFFUSION_PRESETS))}",
    )
    parser.add_argument("--save_root", default="checkpoints/ablation")
    parser.add_argument("--runs_root", default="runs/ablation")
    parser.add_argument("--extra_args", default="", help="Extra arguments appended to each train_diffusion command.")
    return parser.parse_args()


def main():
    args = parse_args()
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in presets if p not in DIFFUSION_PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Available: {sorted(DIFFUSION_PRESETS)}")

    train_script = Path(__file__).resolve().with_name("train_diffusion.py")
    extra = args.extra_args.split() if args.extra_args.strip() else []

    for preset in presets:
        cmd = [
            sys.executable,
            str(train_script),
            "--preset", preset,
            "--experiment_name", preset,
            "--save_dir", args.save_root,
            "--runs_dir", args.runs_root,
            *extra,
        ]
        print(f"[sweep] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
