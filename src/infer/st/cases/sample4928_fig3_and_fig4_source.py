# Combined source bundle for sample 4928 Figure 3 and Figure 4
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os

sample_idx = 4928

# Sample 4928 table
thetas_full = np.array([-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40], dtype=float)
lams_full = np.array([800,850,900,950,1000,1050,1100,1150,1200,1250,1300], dtype=float)

T_full = np.array([
    [0.5357,0.6300,0.5216,0.5736,0.5404,0.6147,0.8309,0.8419,0.7459,0.8419,0.8312,0.6144,0.5405,0.5737,0.5215,0.6299,0.5358],
    [0.3694,0.3792,0.3692,0.2595,0.3144,0.6035,0.7507,0.8620,0.9157,0.8622,0.7509,0.6033,0.3145,0.2596,0.3692,0.3792,0.3694],
    [0.8958,0.9391,0.8784,0.7634,0.6252,0.4684,0.3608,0.2794,0.1873,0.2798,0.3609,0.4681,0.6253,0.7637,0.8784,0.9391,0.8958],
    [0.1973,0.2875,0.3834,0.5254,0.6518,0.8803,0.3845,0.5846,0.8100,0.5845,0.3828,0.8804,0.6518,0.5253,0.3834,0.2874,0.1972],
    [0.8299,0.8224,0.9157,0.9463,0.9691,0.9720,0.9646,0.9573,0.9545,0.9573,0.9647,0.9720,0.9691,0.9465,0.9160,0.8238,0.8293],
    [0.4499,0.3443,0.1537,0.1976,0.8668,0.5097,0.1252,0.0113,0.0490,0.0109,0.1262,0.5091,0.8667,0.1970,0.1536,0.3444,0.4499],
    [0.5017,0.3254,0.1768,0.0678,0.0014,0.0337,0.0323,0.0093,0.0056,0.0091,0.0324,0.0337,0.0016,0.0676,0.1767,0.3252,0.5014],
    [0.6180,0.3746,0.1987,0.0832,0.0128,0.0250,0.0399,0.0425,0.0420,0.0424,0.0398,0.0247,0.0128,0.0833,0.1990,0.3744,0.6181],
    [0.7159,0.8341,0.9503,0.9703,0.6590,0.2809,0.0714,0.0226,0.0497,0.0225,0.0713,0.2806,0.6597,0.9703,0.9503,0.8339,0.7159],
    [0.8919,0.9091,0.9235,0.9374,0.9535,0.9727,0.9729,0.7940,0.5566,0.7938,0.9730,0.9727,0.9535,0.9374,0.9234,0.9090,0.8919],
    [0.9872,0.9871,0.9862,0.9840,0.9756,0.9136,0.0301,0.7813,0.8477,0.7815,0.0294,0.9139,0.9756,0.9840,0.9862,0.9871,0.9873]
], dtype=float)

lambda0_nm = 1100.0
theta_min, theta_max = -20.0, 20.0
lam_min, lam_max = 1050.0, 1150.0

theta_mask = (thetas_full >= theta_min) & (thetas_full <= theta_max)
lam_mask = (lams_full >= lam_min) & (lams_full <= lam_max)
thetas = thetas_full[theta_mask]
lams = lams_full[lam_mask]
T_sub = T_full[np.ix_(lam_mask, theta_mask)]
interp_T = RegularGridInterpolator((lams, thetas), T_sub, bounds_error=False, fill_value=0.0)

c_km_s = 299792.458
T0_fs = lambda0_nm / 299.792458

def T_raw_pu(p, u):
    p = np.asarray(p)
    u = np.asarray(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = lambda0_nm / (1.0 + u)
        ratio = p / (1.0 + u)
    mask = (
        np.isfinite(lam) & np.isfinite(ratio)
        & (lam >= lam_min) & (lam <= lam_max)
        & (ratio >= np.sin(np.deg2rad(theta_min))) & (ratio <= np.sin(np.deg2rad(theta_max)))
    )
    theta = np.rad2deg(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    pts = np.stack([lam, theta], axis=-1)
    out = interp_T(pts)
    return np.where(mask, out, 0.0)

u_min = lambda0_nm / lam_max - 1.0
u_max = lambda0_nm / lam_min - 1.0
p_support = max(abs(np.sin(np.deg2rad(theta_min)) * (1 + u_min)),
                abs(np.sin(np.deg2rad(theta_max)) * (1 + u_max)))
u_scale = max(abs(u_min), abs(u_max))

def T_ideal_pu(p, u):
    p = np.asarray(p)
    u = np.asarray(u)
    support = (
        (u >= u_min) & (u <= u_max)
        & (p >= np.sin(np.deg2rad(theta_min)) * (1 + u))
        & (p <= np.sin(np.deg2rad(theta_max)) * (1 + u))
    )
    T = (p / p_support) ** 2 * (u / u_scale) ** 2
    return np.where(support, T, 0.0)

Nx = 1024
Nt = 4096
x = np.linspace(-160.0, 160.0, Nx, endpoint=False)
t = np.linspace(0.0, 1800.0, Nt, endpoint=False)
dx = x[1] - x[0]
dt = t[1] - t[0]

p_fft = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
u_fft = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))
P_fft, U_fft = np.meshgrid(p_fft, u_fft)

T_grid_raw = T_raw_pu(P_fft.ravel(), U_fft.ravel()).reshape(U_fft.shape)
T_grid_ideal = T_ideal_pu(P_fft.ravel(), U_fft.ravel()).reshape(U_fft.shape)

segments = [(-85.0, -25.0), (-10.0, 35.0), (65.0, 85.0)]
Sx = np.zeros_like(x)
for a, b in segments:
    Sx[(x >= a) & (x <= b)] = 1.0

At = np.zeros_like(t)
At[(t >= 200.0) & (t <= 1600.0)] = 1.0
input1 = At[:, None] * Sx[None, :]

F1 = np.fft.fftshift(np.fft.fft2(input1))
E1_raw = np.fft.ifft2(np.fft.ifftshift(F1 * T_grid_raw))
E1_ideal = np.fft.ifft2(np.fft.ifftshift(F1 * T_grid_ideal))
I1_raw = np.abs(E1_raw) ** 2
I1_ideal = np.abs(E1_ideal) ** 2

def width_profile(tt):
    w = np.full_like(tt, 120.0)
    mask = (tt >= 200.0) & (tt < 1000.0)
    w[mask] = 120.0 - (80.0 / 800.0) * (tt[mask] - 200.0)
    mask = (tt >= 1000.0) & (tt < 1500.0)
    w[mask] = 40.0 + (80.0 / 500.0) * (tt[mask] - 1000.0)
    return w

w_t = width_profile(t)
input2 = (np.abs(x[None, :]) <= (w_t[:, None] / 2.0)).astype(float)

F2 = np.fft.fftshift(np.fft.fft2(input2))
E2_raw = np.fft.ifft2(np.fft.ifftshift(F2 * T_grid_raw))
E2_ideal = np.fft.ifft2(np.fft.ifftshift(F2 * T_grid_ideal))
I2_raw = np.abs(E2_raw) ** 2
I2_ideal = np.abs(E2_ideal) ** 2

def save_panel(data, path, title, vmax=None):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[x.min(), x.max(), t.min(), t.max()],
        vmax=vmax,
    )
    plt.xlabel("x / λ0")
    plt.ylabel("t / T0")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_lambda_theta_panel(data, path, title, lambdas_dense, thetas_dense):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data,
        cmap="RdBu_r",
        aspect="auto",
        origin="lower",
        extent=[thetas_dense.min(), thetas_dense.max(), lambdas_dense.min(), lambdas_dense.max()],
        interpolation="bicubic",
    )
    plt.xlabel("theta (deg)")
    plt.ylabel("lambda (nm)")
    plt.title(title)
    plt.colorbar(im, label="T")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_structure_panel(data, path, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grouped_overview(path, x_fig3, x_fig4, t_axis, rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    col_titles = ["Input", "Ideal", "Sample"]
    x_axes = [x_fig3, x_fig3, x_fig3, x_fig4, x_fig4, x_fig4]
    flat_axes = axes.ravel()

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=16, pad=10)

    for idx, data_row in enumerate(rows):
        for jdx, data in enumerate(data_row):
            ax = axes[idx, jdx]
            x_local = x_axes[idx * 3 + jdx]
            vmax = 1.0 if jdx == 0 else np.quantile(data, 0.999)
            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=[x_local.min(), x_local.max(), t_axis.min(), t_axis.max()],
                vmax=vmax,
            )
            ax.set_xlabel("x / λ0")
            ax.set_ylabel("t / T0")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

outdir = "."

thetas_dense = np.linspace(theta_min, theta_max, 401)
lambdas_dense = np.linspace(lam_min, lam_max, 401)
Ll, Th = np.meshgrid(lambdas_dense, thetas_dense, indexing="ij")
pts_dense = np.stack([Ll.ravel(), Th.ravel()], axis=-1)
T_lambda_theta_dense = interp_T(pts_dense).reshape(Ll.shape)
u_dense = lambda0_nm / Ll - 1.0
p_dense = (1.0 + u_dense) * np.sin(np.deg2rad(Th))
T_ideal_lambda_theta_dense = T_ideal_pu(p_dense, u_dense)

bundle = np.load("data/train_data_20000.npz")
structure4928 = bundle["structures"][sample_idx].astype(float)

save_lambda_theta_panel(
    T_lambda_theta_dense,
    os.path.join(outdir, "sample4928_lambda_theta_window_interpolated.png"),
    "Sample 4928 interpolated T(lambda, theta)",
    lambdas_dense,
    thetas_dense,
)
save_lambda_theta_panel(
    T_ideal_lambda_theta_dense,
    os.path.join(outdir, "sample4928_lambda_theta_window_ideal.png"),
    "Sample 4928 ideal T(lambda, theta)",
    lambdas_dense,
    thetas_dense,
)
save_structure_panel(
    structure4928,
    os.path.join(outdir, "sample4928_structure.png"),
    "Sample 4928 structure",
)

save_panel(input1, os.path.join(outdir, "sample4928_fig3a_input_fixed_segments.png"),
           "Fig. 3(a)-style input: fixed segments with time switching")
save_panel(I1_raw, os.path.join(outdir, "sample4928_fig3b_raw_output_fixed_segments.png"),
           "Fig. 3(b)-style raw output: sample 4928", vmax=np.quantile(I1_raw, 0.999))
save_panel(I1_ideal, os.path.join(outdir, "sample4928_fig3c_ideal_output_fixed_segments.png"),
           "Fig. 3(c)-style ideal 2nd-order output", vmax=np.quantile(I1_ideal, 0.999))
save_panel(input2, os.path.join(outdir, "sample4928_fig3d_input_variable_width.png"),
           "Fig. 3(d)-style input: single segment with variable width")
save_panel(I2_raw, os.path.join(outdir, "sample4928_fig3e_raw_output_variable_width.png"),
           "Fig. 3(e)-style raw output: sample 4928", vmax=np.quantile(I2_raw, 0.999))
save_panel(I2_ideal, os.path.join(outdir, "sample4928_fig3f_ideal_output_variable_width.png"),
           "Fig. 3(f)-style ideal 2nd-order output", vmax=np.quantile(I2_ideal, 0.999))

print(f"T0 = {T0_fs:.3f} fs")
print(f"Raw I1 max = {I1_raw.max():.6f}")
print(f"Ideal I1 max = {I1_ideal.max():.6f}")
print(f"Raw I2 max = {I2_raw.max():.6f}")
print(f"Ideal I2 max = {I2_ideal.max():.6f}")


# ----- Figure 4 -----


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

thetas_full = np.array([-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40], dtype=float)
lams_full = np.array([800,850,900,950,1000,1050,1100,1150,1200,1250,1300], dtype=float)

T_full = np.array([
    [0.5357,0.6300,0.5216,0.5736,0.5404,0.6147,0.8309,0.8419,0.7459,0.8419,0.8312,0.6144,0.5405,0.5737,0.5215,0.6299,0.5358],
    [0.3694,0.3792,0.3692,0.2595,0.3144,0.6035,0.7507,0.8620,0.9157,0.8622,0.7509,0.6033,0.3145,0.2596,0.3692,0.3792,0.3694],
    [0.8958,0.9391,0.8784,0.7634,0.6252,0.4684,0.3608,0.2794,0.1873,0.2798,0.3609,0.4681,0.6253,0.7637,0.8784,0.9391,0.8958],
    [0.1973,0.2875,0.3834,0.5254,0.6518,0.8803,0.3845,0.5846,0.8100,0.5845,0.3828,0.8804,0.6518,0.5253,0.3834,0.2874,0.1972],
    [0.8299,0.8224,0.9157,0.9463,0.9691,0.9720,0.9646,0.9573,0.9545,0.9573,0.9647,0.9720,0.9691,0.9465,0.9160,0.8238,0.8293],
    [0.4499,0.3443,0.1537,0.1976,0.8668,0.5097,0.1252,0.0113,0.0490,0.0109,0.1262,0.5091,0.8667,0.1970,0.1536,0.3444,0.4499],
    [0.5017,0.3254,0.1768,0.0678,0.0014,0.0337,0.0323,0.0093,0.0056,0.0091,0.0324,0.0337,0.0016,0.0676,0.1767,0.3252,0.5014],
    [0.6180,0.3746,0.1987,0.0832,0.0128,0.0250,0.0399,0.0425,0.0420,0.0424,0.0398,0.0247,0.0128,0.0833,0.1990,0.3744,0.6181],
    [0.7159,0.8341,0.9503,0.9703,0.6590,0.2809,0.0714,0.0226,0.0497,0.0225,0.0713,0.2806,0.6597,0.9703,0.9503,0.8339,0.7159],
    [0.8919,0.9091,0.9235,0.9374,0.9535,0.9727,0.9729,0.7940,0.5566,0.7938,0.9730,0.9727,0.9535,0.9374,0.9234,0.9090,0.8919],
    [0.9872,0.9871,0.9862,0.9840,0.9756,0.9136,0.0301,0.7813,0.8477,0.7815,0.0294,0.9139,0.9756,0.9840,0.9862,0.9871,0.9873]
], dtype=float)

lambda0_nm = 1100.0
theta_min, theta_max = -20.0, 20.0
lam_min, lam_max = 1050.0, 1150.0

theta_mask = (thetas_full >= theta_min) & (thetas_full <= theta_max)
lam_mask = (lams_full >= lam_min) & (lams_full <= lam_max)
thetas = thetas_full[theta_mask]
lams = lams_full[lam_mask]
T_sub = T_full[np.ix_(lam_mask, theta_mask)]
interp_T = RegularGridInterpolator((lams, thetas), T_sub, bounds_error=False, fill_value=0.0)

c_km_s = 299792.458
T0_fs = lambda0_nm / 299.792458

def T_raw_pu(p, u):
    p = np.asarray(p)
    u = np.asarray(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = lambda0_nm / (1.0 + u)
        ratio = p / (1.0 + u)
    mask = (
        np.isfinite(lam) & np.isfinite(ratio)
        & (lam >= lam_min) & (lam <= lam_max)
        & (ratio >= np.sin(np.deg2rad(theta_min))) & (ratio <= np.sin(np.deg2rad(theta_max)))
    )
    theta = np.rad2deg(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    pts = np.stack([lam, theta], axis=-1)
    out = interp_T(pts)
    return np.where(mask, out, 0.0)

def interp_at(lam, theta):
    return float(interp_T([[lam, theta]])[0])

u_min = lambda0_nm / lam_max - 1.0
u_max = lambda0_nm / lam_min - 1.0
p_support = max(abs(np.sin(np.deg2rad(theta_min)) * (1 + u_min)),
                abs(np.sin(np.deg2rad(theta_max)) * (1 + u_max)))
u_scale = max(abs(u_min), abs(u_max))

p_axis = np.linspace(np.sin(np.deg2rad(theta_min)) * (1 + u_min),
                     np.sin(np.deg2rad(theta_max)) * (1 + u_max), 2600)
betas = np.linspace(0.001, 0.15, 700)

def raw_overlap(beta):
    return np.trapz(T_raw_pu(p_axis, -beta * p_axis) ** 2, p_axis)


def T_ideal_pu(p, u):
    p = np.asarray(p)
    u = np.asarray(u)
    support = (
        (u >= u_min) & (u <= u_max)
        & (p >= np.sin(np.deg2rad(theta_min)) * (1 + u))
        & (p <= np.sin(np.deg2rad(theta_max)) * (1 + u))
    )
    T = (p / p_support) ** 2 * (u / u_scale) ** 2
    return np.where(support, T, 0.0)

overlaps = np.array([raw_overlap(beta) for beta in betas])
beta0 = float(betas[np.argmax(overlaps)])
v0_km_s = beta0 * c_km_s

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

def center_position(tarr, beta0_local):
    xc = np.zeros_like(tarr)
    mask = tarr < t_static_end
    xc[mask] = x0

    mask = (tarr >= t_static_end) & (tarr < t_v05_end)
    tau = tarr[mask] - t_static_end
    xc[mask] = x0 + 0.5 * beta0_local * tau
    x1 = x0 + 0.5 * beta0_local * (t_v05_end - t_static_end)

    mask = (tarr >= t_v05_end) & (tarr < t_v1_end)
    tau = tarr[mask] - t_v05_end
    xc[mask] = x1 + beta0_local * tau
    x2 = x1 + beta0_local * (t_v1_end - t_v05_end)

    mask = (tarr >= t_v1_end) & (tarr < t_v15_end)
    tau = tarr[mask] - t_v1_end
    xc[mask] = x2 + 1.5 * beta0_local * tau
    x3 = x2 + 1.5 * beta0_local * (t_v15_end - t_v1_end)

    mask = (tarr >= t_v15_end) & (tarr < t_stop_end)
    tau = tarr[mask] - t_v15_end
    Tdec = t_stop_end - t_v15_end
    xc[mask] = x3 + 1.5 * beta0_local * (tau - tau**2 / (2.0 * Tdec))
    x4 = x3 + 1.5 * beta0_local * (Tdec - Tdec / 2.0)

    mask = tarr >= t_stop_end
    xc[mask] = x4
    return xc

xc = center_position(t, beta0)
input_xt = (np.abs(x[None, :] - xc[:, None]) <= width_lambda0 / 2.0).astype(float)

p_fft = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
u_fft = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))
P_fft, U_fft = np.meshgrid(p_fft, u_fft)
T_grid = T_raw_pu(P_fft.ravel(), U_fft.ravel()).reshape(U_fft.shape)
T_grid_ideal = T_ideal_pu(P_fft.ravel(), U_fft.ravel()).reshape(U_fft.shape)

F_in = np.fft.fftshift(np.fft.fft2(input_xt))
F_out = F_in * T_grid
E_out = np.fft.ifft2(np.fft.ifftshift(F_out))
amp_out = np.abs(E_out)
int_out = amp_out ** 2
F_out_ideal = F_in * T_grid_ideal
E_out_ideal = np.fft.ifft2(np.fft.ifftshift(F_out_ideal))
int_out_ideal = np.abs(E_out_ideal) ** 2

u_plot = np.linspace(u_min, u_max, 260)
p_min = np.sin(np.deg2rad(theta_min)) * (1 + u_min)
p_max = np.sin(np.deg2rad(theta_max)) * (1 + u_max)
p_plot = np.linspace(p_min, p_max, 320)
Pp, Uu = np.meshgrid(p_plot, u_plot)
T_plot = T_raw_pu(Pp.ravel(), Uu.ravel()).reshape(Uu.shape)

velocities_km_s = np.logspace(2.0, 4.9, 220)
raw_response = np.array([raw_overlap(v / c_km_s) for v in velocities_km_s])
raw_response_norm = raw_response / raw_response.max()

center_val = interp_at(lambda0_nm, 0.0)
corner_vals = {
    "(1050,-20)": interp_at(1050.0, -20.0),
    "(1050,20)": interp_at(1050.0, 20.0),
    "(1150,-20)": interp_at(1150.0, -20.0),
    "(1150,20)": interp_at(1150.0, 20.0),
}

tf_path = "fig4_sample4928_1100pm20deg_pm50nm_tf.png"
amp_path = "fig4_sample4928_1100pm20deg_pm50nm_output_amplitude.png"
int_path = "fig4_sample4928_1100pm20deg_pm50nm_output_intensity.png"
vel_path = "fig4_sample4928_1100pm20deg_pm50nm_velocity_response.png"
grouped_path = "sample4928_grouped_overview.png"

plt.figure(figsize=(6, 5))
im = plt.imshow(
    T_plot,
    cmap="RdBu_r",
    aspect="auto",
    origin="lower",
    extent=[p_plot.min(), p_plot.max(), u_plot.min(), u_plot.max()],
)
plt.xlabel("kx / k0")
plt.ylabel("Ω / ω0")
plt.title("Sample 4928 raw TF: 1100 nm center, ±20°, 1050–1150 nm")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(tf_path, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
im = plt.imshow(
    amp_out,
    aspect="auto",
    origin="lower",
    extent=[x.min(), x.max(), t.min(), t.max()],
    vmax=np.quantile(amp_out, 0.999),
)
plt.xlabel("x / λ0")
plt.ylabel("t / T0")
plt.title("Sample 4928 output amplitude |E_out|")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(amp_path, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
im = plt.imshow(
    int_out,
    aspect="auto",
    origin="lower",
    extent=[x.min(), x.max(), t.min(), t.max()],
    vmax=np.quantile(int_out, 0.999),
)
plt.xlabel("x / λ0")
plt.ylabel("t / T0")
plt.title("Sample 4928 output intensity |E_out|^2")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(int_path, dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(velocities_km_s, raw_response_norm)
plt.axvline(v0_km_s)
plt.scatter([v0_km_s], [1.0])
plt.xscale("log")
plt.xlabel("Velocity [km/s]")
plt.ylabel("Normalized raw response")
plt.title("Sample 4928 velocity response")
plt.tight_layout()
plt.savefig(vel_path, dpi=200, bbox_inches="tight")
plt.close()

save_grouped_overview(
    grouped_path,
    x,
    x,
    t,
    [
        (input2, I2_ideal, I2_raw),
        (input_xt, int_out_ideal, int_out),
    ],
)

print(f"lambda0 = {lambda0_nm:.1f} nm")
print(f"theta window = [{theta_min:.0f}, {theta_max:.0f}] deg")
print(f"lambda window = [{lam_min:.0f}, {lam_max:.0f}] nm")
print(f"u window = [{u_min:.5f}, {u_max:.5f}]")
print(f"Estimated beta0 = {beta0:.5f}")
print(f"Estimated v0 = {v0_km_s:.1f} km/s")
print(f"T0 = {T0_fs:.3f} fs")
print(f"Amplitude max = {amp_out.max():.6f}")
print(f"Intensity max = {int_out.max():.6f}")
print(f"Ideal intensity max = {int_out_ideal.max():.6f}")
print(f"Center T(1100,0) = {center_val:.4f}")
print("Corner values:")
for k, v in corner_vals.items():
    print(f"  {k}: {v:.4f}")
