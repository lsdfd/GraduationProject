# -*- coding: utf-8 -*-
"""Forward Surrogate Model — 3D feature-map style architecture diagram"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
PAL = {
    "input":   ("#D5E8D4", "#82B366"),   # green
    "stem":    ("#DAE8FC", "#6C8EBF"),   # blue
    "enc":     ("#1A3A5C", "#4A90D9"),   # dark blue
    "bottle":  ("#7B4F00", "#D4A017"),   # amber
    "pool":    ("#555555", "#999999"),   # gray
    "head":    ("#0D4F3C", "#27AE60"),   # teal
    "out_tpp": ("#7B1A1A", "#E74C3C"),   # red
    "out_tss": ("#4A1A7B", "#9B59B6"),   # purple
}
BG = "#F8F9FA"
ARROW_C = "#2C3E50"
LABEL_C = "#1C2833"

# ── 3-D block drawing ─────────────────────────────────────────────────────────
SKEW_X = 0.35   # horizontal skew for top/right face
SKEW_Y = 0.18   # vertical skew

def draw_3d_block(ax, cx, cy, w, h, d,
                  face_color, edge_color, alpha=1.0, zorder=3):
    """
    Draw a 3-D rectangular block centred at (cx, cy).
    w = width (x), h = height (y), d = depth (visual thickness).
    Returns (left_x, right_x, top_y, bottom_y) of the front face.
    """
    sx, sy = d * SKEW_X, d * SKEW_Y   # skew offsets

    # front face corners
    fl = cx - w / 2;  fr = cx + w / 2
    fb = cy - h / 2;  ft = cy + h / 2

    # front face
    front = plt.Polygon(
        [[fl, fb], [fr, fb], [fr, ft], [fl, ft]],
        closed=True, facecolor=face_color, edgecolor=edge_color,
        linewidth=0.9, alpha=alpha, zorder=zorder)
    ax.add_patch(front)

    # top face
    top = plt.Polygon(
        [[fl, ft], [fr, ft],
         [fr + sx, ft + sy], [fl + sx, ft + sy]],
        closed=True,
        facecolor=_lighten(face_color, 0.35),
        edgecolor=edge_color, linewidth=0.9, alpha=alpha, zorder=zorder)
    ax.add_patch(top)

    # right face
    right = plt.Polygon(
        [[fr, fb], [fr + sx, fb + sy],
         [fr + sx, ft + sy], [fr, ft]],
        closed=True,
        facecolor=_darken(face_color, 0.25),
        edgecolor=edge_color, linewidth=0.9, alpha=alpha, zorder=zorder)
    ax.add_patch(right)

    return fl, fr, fb, ft


def _lighten(hex_color, amount):
    r, g, b = _hex2rgb(hex_color)
    r = min(1.0, r + amount); g = min(1.0, g + amount); b = min(1.0, b + amount)
    return _rgb2hex(r, g, b)

def _darken(hex_color, amount):
    r, g, b = _hex2rgb(hex_color)
    r = max(0.0, r - amount); g = max(0.0, g - amount); b = max(0.0, b - amount)
    return _rgb2hex(r, g, b)

def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def _rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def arrow(ax, x1, x2, y, color=ARROW_C, lw=1.4):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=10)

def label_below(ax, cx, y, text, fontsize=7.2, color="#555"):
    ax.text(cx, y, text, ha="center", va="top",
            fontsize=fontsize, color=color)

def label_above(ax, cx, y, text, fontsize=7.5, bold=False):
    ax.text(cx, y, text, ha="center", va="bottom",
            fontsize=fontsize, color=LABEL_C,
            fontweight="bold" if bold else "normal")


# ── figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 7))
ax.set_xlim(-0.5, 20.5)
ax.set_ylim(-1.2, 6.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

CY = 3.0   # vertical centre of all blocks

# ── block specs: (cx, w, h, d, palette_key, top_label, bot_label) ────────────
# w  = visual width  (proportional to spatial size, log scale looks better)
# h  = visual height (proportional to channels)
# d  = depth (constant for clean look)

blocks = [
    # cx     w     h     d    key       top_label          bot_label
    ( 1.0,  0.55, 2.80, 0.55, "input",  "Input",           "1×64×64"),
    ( 2.5,  0.55, 2.80, 0.55, "stem",   "Stem",            "32×64×64"),
    ( 4.0,  0.50, 3.20, 0.55, "enc",    "Enc1",            "64×32×32"),
    ( 5.4,  0.45, 3.60, 0.55, "enc",    "Enc2",            "128×16×16"),
    ( 6.7,  0.40, 4.00, 0.55, "enc",    "Enc3",            "256×8×8"),
    ( 7.9,  0.35, 4.00, 0.55, "enc",    "Enc4",            "256×4×4"),
    ( 9.4,  0.50, 4.00, 0.55, "bottle", "Bottleneck",      "256×4×4\n+Attention"),
    (11.0,  0.40, 3.20, 0.55, "pool",   "AvgPool",         "256×11×17"),
    (12.5,  0.50, 3.20, 0.55, "head",   "Spectral\nHead",  "128→2\n×11×17"),
]

block_rights = []
for (cx, w, h, d, key, top, bot) in blocks:
    fc, ec = PAL[key]
    fl, fr, fb, ft = draw_3d_block(ax, cx, CY, w, h, d, fc, ec)
    block_rights.append((cx, fr, fl, ft, fb))

    # top label (above block)
    label_above(ax, cx + d*SKEW_X/2, ft + d*SKEW_Y + 0.12, top,
                fontsize=8, bold=True)
    # bottom label (below block)
    label_below(ax, cx, fb - 0.15, bot, fontsize=6.8)

# ── arrows between blocks ─────────────────────────────────────────────────────
for i in range(len(blocks) - 1):
    cx_cur, fr_cur, *_ = block_rights[i]
    cx_nxt, fr_nxt, fl_nxt, *_ = block_rights[i+1]
    arrow(ax, fr_cur + 0.04, fl_nxt - 0.04, CY)

# stride labels on encoder arrows
stride_xs = [
    (block_rights[1][1] + block_rights[2][2]) / 2,
    (block_rights[2][1] + block_rights[3][2]) / 2,
    (block_rights[3][1] + block_rights[4][2]) / 2,
    (block_rights[4][1] + block_rights[5][2]) / 2,
]
for sx in stride_xs:
    ax.text(sx, CY + 0.28, "÷2", ha="center", fontsize=7,
            color="#E74C3C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none"))

# ── output heatmaps ───────────────────────────────────────────────────────────
theta = np.linspace(-40, 40, 17)
lam   = np.linspace(800, 1300, 11)

for i, (title, cmap, cx_off) in enumerate([
    ("tpp_mag", "RdYlBu_r", 15.0),
    ("tss_mag", "PuRd",     17.8),
]):
    data = np.sin(np.deg2rad(theta[None, :]))**2 * \
           np.linspace(0.3, 0.9, 11)[:, None]
    if i == 1:
        data = data * 0.6 + 0.1 * np.random.rand(11, 17)

    # inset axes
    iax = ax.inset_axes([(cx_off - 0.9)/21, (CY - 1.5)/7.7,
                          2.2/21, 3.0/7.7])
    im = iax.imshow(data, aspect="auto", cmap=cmap,
                    vmin=0, vmax=1, origin="upper")
    iax.set_xticks([0, 8, 16])
    iax.set_xticklabels(["-40°", "0°", "40°"], fontsize=6)
    iax.set_yticks([0, 5, 10])
    iax.set_yticklabels(["800", "1050", "1300"], fontsize=6)
    iax.set_xlabel("Angle (°)", fontsize=6.5, labelpad=2)
    iax.set_ylabel("λ (nm)", fontsize=6.5, labelpad=2)
    iax.set_title(title, fontsize=8.5, fontweight="bold", pad=4,
                  color=PAL["out_tpp"][1] if i == 0 else PAL["out_tss"][1])
    plt.colorbar(im, ax=iax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5.5)
    for sp in iax.spines.values():
        sp.set_linewidth(0.8)

    # arrow from head to heatmap
    arrow(ax, block_rights[-1][1] + 0.04 + (0 if i == 0 else 2.8),
          cx_off - 1.15, CY, lw=1.2)

    # label below
    ax.text(cx_off + 0.2, CY - 1.85, "[11 × 17]",
            ha="center", fontsize=7, color="#555")

# bracket for outputs
ax.annotate("", xy=(14.6, CY + 2.1), xytext=(14.6, CY - 1.6),
            arrowprops=dict(arrowstyle="-", color="#999", lw=1.2,
                            connectionstyle="arc3,rad=0"))
ax.text(14.75, CY + 0.25, "Output\n[2×11×17]",
        ha="left", va="center", fontsize=8, color=LABEL_C, fontweight="bold")

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(10.0, 6.2,
        "Forward Surrogate Model Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=LABEL_C)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=PAL["stem"][0],   edgecolor=PAL["stem"][1],   label="Stem"),
    mpatches.Patch(facecolor=PAL["enc"][0],    edgecolor=PAL["enc"][1],    label="Encoder ×4  (stride-2 ResBlocks)"),
    mpatches.Patch(facecolor=PAL["bottle"][0], edgecolor=PAL["bottle"][1], label="Bottleneck + Self-Attention"),
    mpatches.Patch(facecolor=PAL["head"][0],   edgecolor=PAL["head"][1],   label="Spectral Head"),
]
ax.legend(handles=legend_items, loc="lower center",
          ncol=4, fontsize=8, framealpha=0.95,
          bbox_to_anchor=(0.46, -0.14),
          edgecolor="#BDC3C7")

out = Path(__file__).resolve().parents[2] / "samples" / "forward_arch.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG)
print(f"saved → {out}")
