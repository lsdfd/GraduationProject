from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _imshow_panel(ax, data: np.ndarray, x: np.ndarray, t: np.ndarray, label: str, title: str, vmax: float, cmap: str = "Reds"):
    im = ax.imshow(
        np.asarray(data, dtype=np.float64),
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(t[0]), float(t[-1])],
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"x [$\lambda$]")
    ax.set_ylabel(r"t [$T$] / $10^4$")
    return im


def _response_intensity(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.complex128)
    return np.abs(arr) ** 2


def _active_x_limits(x: np.ndarray, maps: list[np.ndarray], pad_frac: float = 0.08) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    active = np.zeros(len(x), dtype=bool)
    for data in maps:
        arr = np.asarray(data)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(x):
            continue
        col_active = np.any(arr > 1e-9, axis=0)
        active |= col_active
    if not np.any(active):
        return float(x[0]), float(x[-1])
    idx = np.where(active)[0]
    i0 = int(idx[0])
    i1 = int(idx[-1])
    span = max(i1 - i0, 1)
    pad = max(int(np.ceil(span * float(pad_frac))), 2)
    i0 = max(i0 - pad, 0)
    i1 = min(i1 + pad, len(x) - 1)
    return float(x[i0]), float(x[i1])


def plot_fig3_triptych(out_path: Path, input_xt: np.ndarray, device_xt: np.ndarray, ideal_xt: np.ndarray, x: np.ndarray, t: np.ndarray, labels: tuple[str, str, str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), constrained_layout=True)
    device_int = _response_intensity(device_xt)
    ideal_int = _response_intensity(ideal_xt)
    response_vmax = max(float(np.max(device_int)), float(np.max(ideal_int)), 1e-18)
    im0 = _imshow_panel(axes[0], input_xt, x, t, "(a)", labels[0], 1.0)
    im1 = _imshow_panel(axes[1], device_int, x, t, "(b)", labels[1] + r" ($|g|^2$)", response_vmax)
    im2 = _imshow_panel(axes[2], ideal_int, x, t, "(c)", labels[2] + r" ($|g|^2$)", response_vmax)
    for ax, im in zip(axes, [im0, im1, im2]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save(fig, out_path)


def plot_fig4_main(out_path: Path, input_xt: np.ndarray, device_xt: np.ndarray, x: np.ndarray, t: np.ndarray) -> None:
    x_span = max(float(x[-1] - x[0]), 1.0)
    zoom_x0, zoom_x1 = _active_x_limits(x, [input_xt, _response_intensity(device_xt)])
    fig_w = min(max(12.0, 7.8 * (x_span / 3000.0)), 22.0)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 6.4), constrained_layout=True)
    device_int = _response_intensity(device_xt)
    vmax = max(float(np.max(device_int)), 1e-18)
    im0 = _imshow_panel(axes[0, 0], input_xt, x, t, "(a)", "Input (full)", 1.0)
    im1 = _imshow_panel(axes[0, 1], device_int, x, t, "(b)", r"Device output (full, $|g|^2$)", vmax)
    im2 = _imshow_panel(axes[1, 0], input_xt, x, t, "(c)", "Input (zoom)", 1.0)
    im3 = _imshow_panel(axes[1, 1], device_int, x, t, "(d)", r"Device output (zoom, $|g|^2$)", vmax)
    axes[1, 0].set_xlim(zoom_x0, zoom_x1)
    axes[1, 1].set_xlim(zoom_x0, zoom_x1)
    for ax, im in zip(axes.ravel(), [im0, im1, im2, im3]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save(fig, out_path)


def plot_fig4_compare(out_path: Path, input_xt: np.ndarray, device_xt: np.ndarray, ideal_xt: np.ndarray, x: np.ndarray, t: np.ndarray) -> None:
    x_span = max(float(x[-1] - x[0]), 1.0)
    device_int = _response_intensity(device_xt)
    ideal_int = _response_intensity(ideal_xt)
    zoom_x0, zoom_x1 = _active_x_limits(x, [input_xt, device_int, ideal_int])
    fig_w = min(max(14.0, 10.8 * (x_span / 3000.0)), 28.0)
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, 6.4), constrained_layout=True)
    response_vmax = max(float(np.max(device_int)), float(np.max(ideal_int)), 1e-18)
    im0 = _imshow_panel(axes[0, 0], input_xt, x, t, "(a)", "Input (full)", 1.0)
    im1 = _imshow_panel(axes[0, 1], device_int, x, t, "(b)", r"Device output (full, $|g|^2$)", response_vmax)
    im2 = _imshow_panel(axes[0, 2], ideal_int, x, t, "(c)", r"Ideal output (full, $|g|^2$)", response_vmax)
    im3 = _imshow_panel(axes[1, 0], input_xt, x, t, "(d)", "Input (zoom)", 1.0)
    im4 = _imshow_panel(axes[1, 1], device_int, x, t, "(e)", r"Device output (zoom, $|g|^2$)", response_vmax)
    im5 = _imshow_panel(axes[1, 2], ideal_int, x, t, "(f)", r"Ideal output (zoom, $|g|^2$)", response_vmax)
    for ax in axes[1]:
        ax.set_xlim(zoom_x0, zoom_x1)
    for ax, im in zip(axes.ravel(), [im0, im1, im2, im3, im4, im5]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save(fig, out_path)


def plot_fig4_frequency_and_velocity(
    out_path: Path,
    kx_over_k0: np.ndarray,
    omega_over_omega0: np.ndarray,
    masks: list[np.ndarray],
    colors: list[str],
    labels: list[str],
    velocities_km_s: np.ndarray,
    responses: np.ndarray,
    v0_km_s: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    ax0, ax1 = axes
    ax0.text(0.02, 0.98, "(c)", transform=ax0.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    for mask, color, label in zip(masks, colors, labels):
        yy, xx = np.where(mask)
        if len(xx) == 0:
            continue
        ax0.scatter(kx_over_k0[xx], omega_over_omega0[yy], s=5, alpha=0.55, c=color, label=label, edgecolors="none")
    ax0.set_xlabel(r"$k_x/k_0$")
    ax0.set_ylabel(r"$\Omega/\omega_0 \times 10^3$")
    ax0.set_title("Fourier support")
    ax0.legend(fontsize=8, loc="upper right")

    ax1.text(0.02, 0.98, "(d)", transform=ax1.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    ax1.plot(velocities_km_s, responses, color="red", lw=1.8)
    ax1.scatter([v0_km_s], [1.0], marker="*", s=140, color="magenta", zorder=3)
    ax1.set_xscale("log")
    ax1.set_ylim(0.0, 1.08)
    ax1.set_xlabel("Velocity [km/s]")
    ax1.set_ylabel("Normalized intensity")
    ax1.set_title("Velocity response")
    ax1.grid(alpha=0.28, which="both")
    _save(fig, out_path)


def plot_transfer_compare(out_path: Path, device_otf: np.ndarray, ideal_otf: np.ndarray, thetas_deg: np.ndarray, lambda_nm: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    extent = [float(thetas_deg[0]), float(thetas_deg[-1]), float(lambda_nm[0]), float(lambda_nm[-1])]
    ims = []
    for ax, img, label in zip(axes, [device_otf, ideal_otf], ["device", "ideal"]):
        vmax = max(float(np.max(np.asarray(img, dtype=np.float64))), 1e-18)
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="turbo", extent=extent, vmin=0.0, vmax=vmax)
        ims.append(im)
        ax.set_title(label)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("lambda (nm)")
    fig.suptitle(title, fontsize=12)
    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    _save(fig, out_path)


def plot_input_spectral_support(
    out_path: Path,
    input_energy: np.ndarray,
    support_mask: np.ndarray,
    kappa: np.ndarray,
    nu_milli: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    support = np.asarray(support_mask, dtype=bool)
    energy = np.asarray(input_energy, dtype=np.float64)
    log_energy = np.log10(np.maximum(energy, 1e-18))
    extent = [float(kappa[0]), float(kappa[-1]), float(nu_milli[0]), float(nu_milli[-1])]

    im0 = axes[0].imshow(log_energy, origin="lower", aspect="auto", extent=extent, cmap="magma", interpolation="nearest")
    axes[0].contour(
        kappa,
        nu_milli,
        support.astype(np.float64),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.0,
    )
    axes[0].set_title("Input envelope spectrum")
    axes[0].set_xlabel(r"$k_x/k_0$")
    axes[0].set_ylabel(r"$\Omega/\omega_0 \times 10^3$")

    masked = np.where(support, log_energy, np.nan)
    im1 = axes[1].imshow(masked, origin="lower", aspect="auto", extent=extent, cmap="magma", interpolation="nearest")
    axes[1].contour(
        kappa,
        nu_milli,
        support.astype(np.float64),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.0,
    )
    axes[1].set_title("Inside device support")
    axes[1].set_xlabel(r"$k_x/k_0$")
    axes[1].set_ylabel(r"$\Omega/\omega_0 \times 10^3$")

    fig.suptitle(title, fontsize=12)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, label="log10 energy")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="log10 energy")
    _save(fig, out_path)


def plot_scale_scan_grid(
    out_path: Path,
    maps: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    panel_prefix: str,
) -> None:
    maps = np.asarray(maps, dtype=np.float64)
    maps = maps ** 2
    n_rows, n_cols = maps.shape[:2]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    vmax = max(float(np.max(maps)), 1e-9)
    last_im = None
    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            last_im = ax.imshow(
                maps[r, c],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="Reds",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{row_labels[r]}\n" + r"t [$T$] / $10^4$")
            else:
                ax.set_ylabel(r"t [$T$] / $10^4$")
            ax.set_xlabel(r"x [$\lambda$]")
            ax.text(
                0.02,
                0.98,
                f"{panel_prefix}{r * n_cols + c + 1}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
    fig.suptitle(title, fontsize=12)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    _save(fig, out_path)
