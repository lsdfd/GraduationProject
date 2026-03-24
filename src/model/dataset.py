import numpy as np
import torch
from torch.utils.data import Dataset


class RCWADataset(Dataset):
    def __init__(self, npz_path, cond_mean=None, cond_std=None, target_lambda=1000.0):
        data = np.load(npz_path)

        structures = data["structures"].astype(np.float32)
        assert structures.ndim == 3, "structures shape must be [N, 64, 64]"

        if "tpp_mag" not in data.files or "tss_mag" not in data.files:
            raise ValueError("Need tpp_mag and tss_mag in npz.")

        tpp = data["tpp_mag"].astype(np.float32)
        tss = data["tss_mag"].astype(np.float32)

        if tpp.ndim == 2 and tss.ndim == 2:
            cond = np.stack([tpp, tss], axis=1).astype(np.float32)  # [N,2,17]
            lambdas = np.asarray([float(data["target_lambda"])], dtype=np.float32) if "target_lambda" in data.files else np.asarray([float(target_lambda)], dtype=np.float32)
            lam_idx = 0
        elif tpp.ndim == 3 and tss.ndim == 3:
            if "lambdas" not in data.files:
                raise ValueError("Need lambdas in npz to select target_lambda row.")
            lambdas = data["lambdas"].astype(np.float32)
            lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
            cond = np.stack([tpp[:, lam_idx, :], tss[:, lam_idx, :]], axis=1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported tpp/tss shape: {tpp.shape}, {tss.shape}")

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
        self.target_lambda = float(lambdas[lam_idx])
        self.target_lambda_idx = lam_idx
        self.lambdas = lambdas

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        x = self.structures[idx]  # [64,64], 0/1
        c = self.cond[idx]        # [C,17]

        x = torch.from_numpy(x).unsqueeze(0).float()  # [1,64,64]
        c = torch.from_numpy(c).float()

        return x, c
