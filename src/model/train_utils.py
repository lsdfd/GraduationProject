import os

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def append_csv(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(map(str, row)) + "\n")


class TrainLogger:
    def __init__(self, name, save_dir, headers):
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, f"{name}_log.csv")
        self.preview_dir = os.path.join(save_dir, f"{name}_preview")
        self.writer = SummaryWriter(f"runs/{name}") if SummaryWriter is not None else None
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
