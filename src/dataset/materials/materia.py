"""Material interpolation for RCWA (self-contained in this repo)."""

from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d


class Material(torch.autograd.Function):
    """Interpolate complex refractive index n+ik from local tabulated data."""

    @staticmethod
    def forward(ctx, material, wavelength, dl=0.005):
        """
        Args:
            material: material name, e.g. 'SiO2'
            wavelength: wavelength in um (float or tensor)
            dl: central-difference step in um
        """
        materials_dir = Path(__file__).resolve().parent / "nk_data"
        data_path = materials_dir / f"{material}.txt"
        if not data_path.exists():
            raise FileNotFoundError(f"Material file not found: {data_path}")

        nk_data = np.loadtxt(data_path)
        n_interp = interp1d(nk_data[:, 0], nk_data[:, 1], kind="cubic")
        k_interp = interp1d(nk_data[:, 0], nk_data[:, 2], kind="cubic")

        if isinstance(wavelength, torch.Tensor):
            wavelength_np = float(wavelength.detach().cpu().item())
            out_device = wavelength.device
        else:
            wavelength_np = float(wavelength)
            out_device = torch.device("cpu")

        def eval_nk(lam):
            if lam < nk_data[0, 0]:
                return nk_data[0, 1] + 1.0j * nk_data[0, 2]
            if lam > nk_data[-1, 0]:
                return nk_data[-1, 1] + 1.0j * nk_data[-1, 2]
            return n_interp(lam) + 1.0j * k_interp(lam)

        nk_value = eval_nk(wavelength_np)
        nk_value_m = eval_nk(wavelength_np - dl)
        nk_value_p = eval_nk(wavelength_np + dl)

        # Cache derivative for backward
        dnk_dl = (nk_value_p - nk_value_m) / (2 * dl)
        ctx.dnk_dl = torch.tensor(dnk_dl, dtype=torch.complex64, device=out_device)

        return torch.tensor(nk_value, dtype=torch.complex64, device=out_device)

    @staticmethod
    def backward(ctx, grad_output):
        grad = 2 * torch.real(torch.conj(grad_output) * ctx.dnk_dl)
        return None, grad, None
