import numpy as np
import torch
from torch.utils.data import Dataset


class RCWADataset(Dataset):
    def __init__(self, npz_path, cond_mean=None, cond_std=None, target_lambda=1000.0):
        data = np.load(npz_path)
        required = {"structures", "tpp_mag", "tss_mag", "target_lambda", "thetas"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{npz_path} 缺少字段: {sorted(missing)}")

        structures = data["structures"].astype(np.float32)
        assert structures.ndim == 3, "structures shape must be [N, 64, 64]"
        tpp = data["tpp_mag"].astype(np.float32)
        tss = data["tss_mag"].astype(np.float32)
        thetas = data["thetas"].astype(np.float32)
        data_lambda = float(data["target_lambda"])
        if abs(data_lambda - float(target_lambda)) > 1e-4:
            raise ValueError(
                f"Dataset target_lambda={data_lambda:.1f} nm 与请求的 {float(target_lambda):.1f} nm 不一致。"
            )
        if tpp.ndim != 2 or tss.ndim != 2:
            raise ValueError(
                f"onelambda 数据集要求 tpp/tss 形状为 [N,17]，实际得到 {tpp.shape} 和 {tss.shape}"
            )
        if tpp.shape != tss.shape:
            raise ValueError(f"tpp/tss shape mismatch: {tpp.shape} vs {tss.shape}")
        if tpp.shape[1] != len(thetas):
            raise ValueError(f"tpp/tss theta 维长度 {tpp.shape[1]} 与 thetas 长度 {len(thetas)} 不一致")
        cond = np.stack([tpp, tss], axis=1).astype(np.float32)  # [N,2,17]

        valid_mask = np.isfinite(cond).all(axis=(1, 2))
        if not np.all(valid_mask):
            structures = structures[valid_mask]
            cond = cond[valid_mask]

        if len(structures) == 0:
            raise ValueError("No valid samples remain after filtering NaN/Inf conditions.")

        if cond_mean is None or cond_std is None:
            cond_mean = cond.mean(axis=0, keepdims=True)  # [1, 2, 17]
            cond_std = cond.std(axis=0, keepdims=True) + 1e-6  # [1, 2, 17]

        self.structures = structures
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = cond_std.astype(np.float32)
        self.cond = (cond - self.cond_mean) / self.cond_std
        self.target_lambda = data_lambda
        self.target_lambda_idx = 0
        self.lambdas = np.asarray([data_lambda], dtype=np.float32)
        self.thetas = thetas

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        x = self.structures[idx]  # [64,64], 0/1
        c = self.cond[idx]        # [C,17]

        x = torch.from_numpy(x).unsqueeze(0).float()  # [1,64,64]
        c = torch.from_numpy(c).float()

        return x, c
