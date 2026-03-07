import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter, binary_opening, binary_closing


# 白色是空气
# ============================================================
# 参数区：主要改这里
# ============================================================

RNG_SEED = None          # None 表示每次随机；改成整数可复现，例如 0
N_COARSE = 16            # 粗网格尺寸
N_FINE = 64              # 最终细网格尺寸

SIGMA1 = 1.6             # 第一次高斯滤波：控制大轮廓是否碎
SIGMA2 = 1.0             # 第二次高斯滤波：控制边界是否圆滑
TARGET_FILL = 0.4      # 目标占空比（材料面积比例），0~1
MIN_FEATURE_PX = 5      # 最小特征尺寸（像素级近似控制），越大越不碎

SAVE_FIG = True          # 是否保存流程图
SAVE_NPY = True          # 是否保存最终64x64数组

# ============================================================
# 输出目录：保存在脚本同目录下的 outputs 文件夹
# ============================================================

SAVE_DIR = Path(__file__).resolve().parent / "outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 基本工具函数
# ============================================================

def normalize(a):
    """把数组归一化到 [0, 1]"""
    a = a.astype(float)
    a = a - a.min()
    m = a.max()
    return a / (m + 1e-12)


def threshold_by_fill(a, fill):
    """
    按目标占空比做二值化
    约定：1 = 材料，0 = 空气
    这里让“值较大”的部分变成材料，以逼近目标占空比 fill
    """
    fill = float(np.clip(fill, 0.0, 1.0))
    if fill <= 0:
        return np.zeros_like(a, dtype=np.uint8)
    if fill >= 1:
        return np.ones_like(a, dtype=np.uint8)

    th = np.quantile(a, 1.0 - fill)
    return (a >= th).astype(np.uint8)


def symmetrize_c4_sigmax(a):
    """
    对灰度场施加 C4 + sigma_x 对称

    说明：
    - C4：旋转 90° 不变
    - sigma_x：关于 x 轴镜像
    - 在图像/矩阵里，默认“上-下翻转 flipud”对应 sigma_x
    - 这里采用 D4 群的 8 个对称副本求平均，最稳妥
    """
    ops = []
    for k in range(4):
        r = np.rot90(a, k)
        ops.append(r)
        ops.append(np.flipud(r))
    return sum(ops) / len(ops)


def symmetrize_binary(b):
    """
    对二值图再次强制对称，避免数值处理带来极小误差
    """
    x = symmetrize_c4_sigmax(b.astype(float))
    return (x >= 0.5).astype(np.uint8)


def clean_binary(b, min_feature_px):
    """
    用形态学开闭运算做一个“最小特征尺寸”的近似约束

    作用：
    - 开运算：去掉小毛刺、小孤岛、太细桥连
    - 闭运算：填掉小孔洞、小裂缝
    - 最后再对称化一次，确保仍满足 C4 + sigma_x

    注意：
    这不是严格数学意义上的“最小线宽测量”，
    而是工程上很常用的近似清理方法。
    """
    k = max(1, int(min_feature_px))
    if k <= 1:
        return symmetrize_binary(b)

    structure = np.ones((k, k), dtype=bool)
    x = b.astype(bool)
    x = binary_opening(x, structure=structure)
    x = binary_closing(x, structure=structure)
    x = symmetrize_binary(x.astype(np.uint8)).astype(bool)
    return x.astype(np.uint8)


def check_symmetry(b):
    """检查最终二值图是否满足 C4 和 sigma_x"""
    c4_ok = np.array_equal(b, np.rot90(b, 1))
    sx_ok = np.array_equal(b, np.flipud(b))
    return c4_ok, sx_ok


# ============================================================
# 主流程：粗网格随机 -> 对称化 -> 插值 -> 两次高斯+二值化
# ============================================================

def generate_structure():
    rng = np.random.default_rng(RNG_SEED)

    # 1) 粗网格随机场
    coarse = rng.standard_normal((N_COARSE, N_COARSE))

    # 2) 施加 C4 + sigma_x 对称
    coarse_sym = symmetrize_c4_sigmax(coarse)

    # 3) 插值到 64x64 细网格
    # order=1 是双线性插值，够简洁稳妥
    fine = zoom(coarse_sym, N_FINE / N_COARSE, order=1)
    fine = normalize(fine)

    # 4) 第一次高斯滤波 + 第一次二值化
    # mode='wrap' 更像周期单胞
    blur1 = gaussian_filter(fine, sigma=SIGMA1, mode="wrap")
    blur1 = normalize(blur1)
    bin1 = threshold_by_fill(blur1, TARGET_FILL)
    bin1 = symmetrize_binary(bin1)

    # 5) 第二次高斯滤波 + 第二次二值化
    blur2 = gaussian_filter(bin1.astype(float), sigma=SIGMA2, mode="wrap")
    blur2 = normalize(blur2)
    final = threshold_by_fill(blur2, TARGET_FILL)

    # 6) 最小特征尺寸清理
    final = clean_binary(final, MIN_FEATURE_PX)

    # 7) 再次对称化，保险
    final = symmetrize_binary(final)

    # 8) 检查
    c4_ok, sx_ok = check_symmetry(final)
    fill_ratio = final.mean()

    return {
        "coarse": coarse,
        "coarse_sym": coarse_sym,
        "fine": fine,
        "blur1": blur1,
        "bin1": bin1,
        "blur2": blur2,
        "final": final,
        "c4_ok": c4_ok,
        "sx_ok": sx_ok,
        "fill_ratio": fill_ratio,
    }


# ============================================================
# 画图：黑色 = 材料，白色 = 空气
# ============================================================

def plot_results(data):
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    axes = axes.ravel()

    imgs = [
        data["coarse"],
        data["coarse_sym"],
        data["fine"],
        data["blur1"],
        data["bin1"],
        data["blur2"],
        data["final"],
    ]

    titles = [
        "1. coarse random field",
        "2. enforce C4 + sigma_x",
        "3. upsample to 64x64",
        "4. first gaussian blur",
        "5. first binarization",
        "6. second gaussian blur",
        "7. final binary structure",
    ]

    for i, (img, title) in enumerate(zip(imgs, titles)):
        # gray_r: 1 -> 黑色，0 -> 白色，因此黑色=材料
        axes[i].imshow(img, cmap="gray_r", interpolation="nearest")
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    axes[7].text(0.05, 0.78, f"C4 symmetry: {data['c4_ok']}", fontsize=11)
    axes[7].text(0.05, 0.58, f"sigma_x symmetry: {data['sx_ok']}", fontsize=11)
    axes[7].text(0.05, 0.38, f"fill ratio: {data['fill_ratio']:.4f}", fontsize=11)
    axes[7].text(0.05, 0.18, "black = material", fontsize=11)
    axes[7].axis("off")

    plt.tight_layout()

    if SAVE_FIG:
        fig_path = SAVE_DIR / "freeform_c4_sigmax_demo.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"流程图已保存到: {fig_path}")

    plt.show()


# ============================================================
# 保存最终数组
# ============================================================

def save_final_array(final):
    if SAVE_NPY:
        npy_path = SAVE_DIR / "final_structure_64x64.npy"
        np.save(npy_path, final)
        print(f"最终数组已保存到: {npy_path}")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    data = generate_structure()
    final = data["final"]

    print("最终数组约定：1 = 材料, 0 = 空气")
    print("显示约定：黑色 = 材料, 白色 = 空气")
    print(f"最终占空比(材料面积比例): {data['fill_ratio']:.4f}")
    print(f"C4 对称: {data['c4_ok']}")
    print(f"sigma_x 对称: {data['sx_ok']}")
    print("调参建议：")
    print("- 结构太碎：增大 SIGMA1 / SIGMA2 / MIN_FEATURE_PX，或减小 N_COARSE")
    print("- 结构太胖：减小 TARGET_FILL")
    print("- 结构太瘦：增大 TARGET_FILL")
    print("- 想更细节：增大 N_COARSE；想更大块：减小 N_COARSE")

    save_final_array(final)
    plot_results(data)