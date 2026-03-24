import numpy as np
import torch
from torch.utils.data import Dataset


class RCWADataset(Dataset):
    def __init__(self, npz_path, cond_mean=None, cond_std=None):
        data = np.load(npz_path)

        structures = data["structures"].astype(np.float32)
        assert structures.ndim == 3, "structures shape must be [N, 64, 64]"

        # Strict new format: two magnitude channels [tpp_mag, tss_mag].
        if "tpp_mag" not in data.files or "tss_mag" not in data.files:
            raise ValueError("Need tpp_mag and tss_mag in npz.")
        cond = np.stack([data["tpp_mag"], data["tss_mag"]], axis=1).astype(np.float32)  # [N,2,11,13]

        # RCWA 失败点会写成 NaN，这里直接丢掉无效样本，避免污染标准化统计。
        valid_mask = np.isfinite(cond).all(axis=(1, 2, 3))
        if not np.all(valid_mask):
            structures = structures[valid_mask]
            cond = cond[valid_mask]

        if len(structures) == 0:
            raise ValueError("No valid samples remain after filtering NaN/Inf conditions.")

        if cond_mean is None or cond_std is None:
            # 按 (channel, lambda, theta) 每个位置独立归一化，而不是全局归一化
            # 这样不同波长/角度的分布差异不会被平均掉
            cond_mean = cond.mean(axis=0, keepdims=True)  # [1, 2, 11, 13]
            cond_std = cond.std(axis=0, keepdims=True) + 1e-6  # [1, 2, 11, 13]

        self.structures = structures
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = cond_std.astype(np.float32)
        self.cond = (cond - self.cond_mean) / self.cond_std

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        x = self.structures[idx]  # [64,64], 0/1
        c = self.cond[idx]        # [C,11,13]

        x = torch.from_numpy(x).unsqueeze(0).float()  # [1,64,64]
        c = torch.from_numpy(c).float()

        return x, c
