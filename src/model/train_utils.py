import os
import time
import json
from pathlib import Path

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def append_csv(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, row)) + "\n")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def prepare_run_dir(base_dir, run_name):
    base = Path(base_dir)
    runs_root = base / f"{run_name}_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / time.strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def update_latest_run(base_dir, run_name, run_dir):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    latest_path = base / f"latest_{run_name}_run.txt"
    latest_path.write_text(str(Path(run_dir).resolve()), encoding="utf-8")
    return latest_path


def resolve_latest_run(base_dir, run_name):
    base = Path(base_dir)
    latest_path = base / f"latest_{run_name}_run.txt"
    if latest_path.exists():
        run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
        if run_dir.exists():
            return run_dir

    runs_root = base / f"{run_name}_runs"
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class TrainLogger:
    def __init__(self, name, save_dir, headers):
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, f"{name}_log.csv")
        self.preview_dir = os.path.join(save_dir, f"{name}_preview")
        self.tb_dir = os.path.join(save_dir, "tensorboard")
        self.writer = SummaryWriter(self.tb_dir) if SummaryWriter is not None else None
        if not os.path.exists(self.csv_path):
            append_csv(self.csv_path, headers)

    def log_scalars(self, epoch, row, scalars):
        append_csv(self.csv_path, row)
        if self.writer is None:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, epoch)

    def save_preview(self, epoch, sample, tag="preview/sample"):
        os.makedirs(self.preview_dir, exist_ok=True)
        np.save(os.path.join(self.preview_dir, f"epoch_{epoch:03d}.npy"), sample.numpy())
        if self.writer is not None:
            self.writer.add_images(tag, sample[:1], epoch)

    def close(self):
        if self.writer is not None:
            self.writer.close()
