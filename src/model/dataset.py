import numpy as np
import torch
from torch.utils.data import Dataset


class RCWADataset(Dataset):
    def __init__(self, npz_path, cond_mean=None, cond_std=None, target_lambda=1000.0):
        data = np.load(npz_path)
        required = {"structures", "tpp_mag", "target_lambda", "thetas"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{npz_path} 缺少字段: {sorted(missing)}")

        structures = data["structures"].astype(np.float32)
        tpp = data["tpp_mag"].astype(np.float32)
        thetas = data["thetas"].astype(np.float32)
        data_lambda = float(data["target_lambda"])

        if structures.ndim != 3 or structures.shape[1:] != (64, 64):
            raise ValueError(f"structures shape must be [N,64,64], got {structures.shape}")
        if abs(data_lambda - float(target_lambda)) > 1e-4:
            raise ValueError(
                f"Dataset target_lambda={data_lambda:.1f} nm 与请求的 {float(target_lambda):.1f} nm 不一致。"
            )
        if tpp.ndim != 2:
            raise ValueError(f"only-tpp 数据集要求 tpp_mag 形状为 [N,17]，实际得到 {tpp.shape}")
        if tpp.shape[1] != len(thetas):
            raise ValueError(f"tpp theta 维长度 {tpp.shape[1]} 与 thetas 长度 {len(thetas)} 不一致")

        valid_mask = np.isfinite(tpp).all(axis=1)
        if not np.all(valid_mask):
            structures = structures[valid_mask]
            tpp = tpp[valid_mask]

        if len(structures) == 0:
            raise ValueError("No valid samples remain after filtering NaN/Inf conditions.")

        if cond_mean is None or cond_std is None:
            cond_mean = tpp.mean(axis=0, keepdims=True)  # [1,17]
            cond_std = tpp.std(axis=0, keepdims=True) + 1e-6  # [1,17]

        self.structures = structures
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = cond_std.astype(np.float32)
        self.cond = ((tpp - self.cond_mean) / self.cond_std).astype(np.float32)
        self.target_lambda = data_lambda
        self.target_lambda_idx = 0
        self.lambdas = np.asarray([data_lambda], dtype=np.float32)
        self.thetas = thetas

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.structures[idx]).unsqueeze(0).float()  # [1,64,64]
        c = torch.from_numpy(self.cond[idx]).float()  # [17]
        return x, c
