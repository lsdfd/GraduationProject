import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
npz_path = ROOT / "data" / "train_data_20000.npz"
sample_idx = 4928
channel_key = "tpp_mag"
data = np.load(npz_path)

lambda0_nm = 1150.0
c_km_s = 299792.458
T0_fs = lambda0_nm / 299.792458

theta_min_deg = -20.0
theta_max_deg = 20.0
lambda_min_nm = 1100.0
lambda_max_nm = 1200.0

all_thetas = np.asarray(data["thetas"], dtype=float)
all_lams = np.asarray(data["lambdas"], dtype=float)
theta_mask = (all_thetas >= theta_min_deg) & (all_thetas <= theta_max_deg)
lams = all_lams[(all_lams >= lambda_min_nm) & (all_lams <= lambda_max_nm)]
thetas = all_thetas[theta_mask]
sample_map = np.asarray(data[channel_key][sample_idx], dtype=float)
T_raw = sample_map[np.ix_((all_lams >= lambda_min_nm) & (all_lams <= lambda_max_nm), theta_mask)].T
interp_T_theta_lambda = RegularGridInterpolator(
    (thetas, lams), T_raw, bounds_error=False, fill_value=0.0
)

def T_pu_raw(p, u):
    p = np.asarray(p)
    u = np.asarray(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = lambda0_nm / (1.0 + u)
        ratio = p / (1.0 + u)
    mask = (
        np.isfinite(lam)
        & np.isfinite(ratio)
        & (lam >= lams.min())
        & (lam <= lams.max())
        & (np.abs(ratio) <= 1.0)
    )
    theta = np.rad2deg(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    pts = np.stack([theta, lam], axis=-1)
    out = interp_T_theta_lambda(pts)
    return np.where(mask, out, 0.0)

T00 = float(T_pu_raw(np.array([0.0]), np.array([0.0]))[0])

def T_pu_corr(p, u):
    """Use the raw interpolated OTF directly, without zero-line correction."""
    return T_pu_raw(p, u)

# Estimate v0 by spectral overlap
p_axis = np.linspace(-0.72, 0.72, 2000)
betas = np.linspace(0.01, 0.35, 500)

def overlap(beta):
    return np.trapezoid(T_pu_corr(p_axis, -beta * p_axis) ** 2, p_axis)

overlaps = np.array([overlap(beta) for beta in betas])
beta0 = float(betas[np.argmax(overlaps)])
v0_km_s = beta0 * c_km_s

# Space-time grid
Nx = 1024
Nt = 4096
x = np.linspace(-150.0, 150.0, Nx, endpoint=False)
t = np.linspace(0.0, 1800.0, Nt, endpoint=False)
dx = x[1] - x[0]
dt = t[1] - t[0]

x0 = -90.0
t_static_end = 200.0
t_v05_end = 700.0
t_v1_end = 1000.0
t_v15_end = 1200.0
t_stop_end = 1500.0
width_lambda0 = 28.0

def center_position(tarr):
    xc = np.zeros_like(tarr)

    mask = tarr < t_static_end
    xc[mask] = x0

    mask = (tarr >= t_static_end) & (tarr < t_v05_end)
    tau = tarr[mask] - t_static_end
    xc[mask] = x0 + 0.5 * beta0 * tau
    x1 = x0 + 0.5 * beta0 * (t_v05_end - t_static_end)

    mask = (tarr >= t_v05_end) & (tarr < t_v1_end)
    tau = tarr[mask] - t_v05_end
    xc[mask] = x1 + beta0 * tau
    x2 = x1 + beta0 * (t_v1_end - t_v05_end)

    mask = (tarr >= t_v1_end) & (tarr < t_v15_end)
    tau = tarr[mask] - t_v1_end
    xc[mask] = x2 + 1.5 * beta0 * tau
    x3 = x2 + 1.5 * beta0 * (t_v15_end - t_v1_end)

    mask = (tarr >= t_v15_end) & (tarr < t_stop_end)
    tau = tarr[mask] - t_v15_end
    Tdec = t_stop_end - t_v15_end
    xc[mask] = x3 + 1.5 * beta0 * (tau - tau**2 / (2.0 * Tdec))
    x4 = x3 + 1.5 * beta0 * (Tdec - Tdec / 2.0)

    mask = tarr >= t_stop_end
    xc[mask] = x4
    return xc

xc = center_position(t)
input_xt = (np.abs(x[None, :] - xc[:, None]) <= width_lambda0 / 2.0).astype(float)

# Filter in Fourier domain
p_fft = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
u_fft = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))
P_fft, U_fft = np.meshgrid(p_fft, u_fft)
T_corr_grid = T_pu_corr(P_fft.ravel(), U_fft.ravel()).reshape(U_fft.shape)

F_in = np.fft.fftshift(np.fft.fft2(input_xt))
F_out = F_in * T_corr_grid
output_xt = np.abs(np.fft.ifft2(np.fft.ifftshift(F_out)))

def moving_bar_uniform(beta, width=28.0, t_span=1200.0):
    Nt2 = 2048
    Nx2 = 1024
    x2 = np.linspace(-150.0, 150.0, Nx2, endpoint=False)
    t2 = np.linspace(0.0, t_span, Nt2, endpoint=False)
    xc2 = beta * (t2 - t2.mean())
    inp2 = (np.abs(x2[None, :] - xc2[:, None]) <= width / 2.0).astype(float)
    F2 = np.fft.fftshift(np.fft.fft2(inp2))
    mag = np.abs(F2)
    p2 = np.fft.fftshift(np.fft.fftfreq(Nx2, d=x2[1] - x2[0]))
    u2 = np.fft.fftshift(np.fft.fftfreq(Nt2, d=t2[1] - t2[0]))
    return p2, u2, (mag >= 0.015 * mag.max()).astype(float)

beta_list = [0.5 * beta0, beta0, 1.5 * beta0]
mask_data = [moving_bar_uniform(beta, width=width_lambda0) for beta in beta_list]

velocities_km_s = np.logspace(2.0, 5.2, 250)
response = np.array([overlap(v / c_km_s) for v in velocities_km_s])
response = response / response.max()

outdir = os.path.dirname(__file__)
suffix = f"_sample{sample_idx}_{channel_key.replace('_mag','')}_th{int(theta_min_deg)}_{int(theta_max_deg)}_lam{int(lambda_min_nm)}_{int(lambda_max_nm)}"
input_path = os.path.join(outdir, f"fig4_repro_input_raw{suffix}.png")
output_path = os.path.join(outdir, f"fig4_repro_output_raw{suffix}.png")
spectral_path = os.path.join(outdir, f"fig4_repro_spectral_overlap_raw{suffix}.png")
velocity_path = os.path.join(outdir, f"fig4_repro_velocity_response_raw{suffix}.png")
curve_csv_path = os.path.join(outdir, f"fig4_repro_velocity_curve_raw{suffix}.csv")

# Plot 1
plt.figure(figsize=(6, 5))
im1 = plt.imshow(
    input_xt,
    aspect="auto",
    origin="lower",
    extent=[x.min(), x.max(), t.min(), t.max()],
)
plt.xlabel("x / λ0")
plt.ylabel("t / T0")
plt.title(f"Approximate Fig. 4(a): input moving 1D segment | sample {sample_idx} {channel_key}")
plt.colorbar(im1)
plt.tight_layout()
plt.savefig(input_path, dpi=200, bbox_inches="tight")
plt.close()

# Plot 2
plt.figure(figsize=(6, 5))
im2 = plt.imshow(
    output_xt,
    aspect="auto",
    origin="lower",
    extent=[x.min(), x.max(), t.min(), t.max()],
    vmax=np.quantile(output_xt, 0.999),
)
plt.xlabel("x / λ0")
plt.ylabel("t / T0")
plt.title(f"Approximate Fig. 4(b): filtered output | sample {sample_idx} {channel_key}")
plt.colorbar(im2)
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

# Plot 3
u_plot = np.linspace(-0.12, 0.12, 260)
p_plot = np.linspace(-0.72, 0.72, 320)
Pp, Uu = np.meshgrid(p_plot, u_plot)
T_plot = T_pu_corr(Pp.ravel(), Uu.ravel()).reshape(Uu.shape)

plt.figure(figsize=(6, 5))
im3 = plt.imshow(
    T_plot,
    aspect="auto",
    origin="lower",
    extent=[p_plot.min(), p_plot.max(), u_plot.min(), u_plot.max()],
)
for p2, u2, mask in mask_data:
    plt.contour(p2, u2, mask, levels=[0.5])
plt.xlabel("kx / k0")
plt.ylabel("Ω / ω0")
plt.title(f"Approximate Fig. 4(c): TF and spectral masks | sample {sample_idx} {channel_key}")
plt.colorbar(im3)
plt.tight_layout()
plt.savefig(spectral_path, dpi=200, bbox_inches="tight")
plt.close()

# Plot 4
plt.figure(figsize=(6, 5))
plt.plot(velocities_km_s, response)
plt.axvline(v0_km_s)
plt.scatter([v0_km_s], [1.0])
plt.xscale("log")
plt.xlabel("Velocity [km/s]")
plt.ylabel("Normalized edge intensity")
plt.title(f"Approximate Fig. 4(d): velocity response | sample {sample_idx} {channel_key}")
plt.tight_layout()
plt.savefig(velocity_path, dpi=200, bbox_inches="tight")
plt.close()

np.savetxt(
    curve_csv_path,
    np.column_stack([velocities_km_s, response]),
    delimiter=",",
    header="velocity_km_s,normalized_edge_intensity",
    comments="",
)

print(f"npz_path = {npz_path}")
print(f"sample_idx = {sample_idx}")
print(f"channel_key = {channel_key}")
print(f"lambda_window = [{lambda_min_nm:.0f}, {lambda_max_nm:.0f}] nm")
print(f"theta_window = [{theta_min_deg:.0f}, {theta_max_deg:.0f}] deg")
print(f"Estimated beta0 = {beta0:.4f}")
print(f"Estimated v0 = {v0_km_s:.1f} km/s")
print(f"T0 = {T0_fs:.3f} fs")
