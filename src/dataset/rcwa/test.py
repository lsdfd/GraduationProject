import torch
# 从rcwa.py导入核心仿真函数
from rcwa import torcwa_simulation

# ===================== 1. 配置论文典型结构参数（核心！对应论文图9/3.4节）=====================
# 硅中空砖超表面-固定参数（论文143行：C4v对称，砖高H固定450nm）
H = 450.0    # 砖高(nm)
# 1250nm二阶微分器-最优结构参数（论文图9(a)：W=152, L=339, P=610）
P = 610.0    # 晶胞周期(nm)
W = 152.0    # 内孔边缘长度(nm)
L_brick = 339.0  # 砖块边缘长度(nm)
# 工作参数（论文3.4节：二阶微分器目标波长1250nm，入射角0°）
lam = 1250.0 # 入射波长(nm)
tet = 0.0    # 入射角(°)，论文重点验证0°入射基础特性
# 材料参数（论文123行：基底SiO2，结构Si，无损耗）
substrate = "SiO2"
structure = "Si"

# ===================== 2. 设备配置（自动CUDA/CPU）=====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"仿真设备: {device}")

# ===================== 3. 生成64x64中空砖结构层（适配RCWA网格，C4v对称）=====================
# 网格分辨率：64x64，单像素尺寸=晶胞周期/64
pixel_size = P / 64
# 计算砖块/内孔在64x64网格中的像素范围（中心对称，C4v）
# 砖块像素范围：从中心向四周扩展L_brick/2
brick_half_pix = int((L_brick / 2) / pixel_size)
# 内孔像素范围：从中心向四周扩展W/2
hole_half_pix = int((W / 2) / pixel_size)
# 初始化64x64结构层：0=空气，1=硅
layer = torch.zeros((64, 64), device=device, dtype=torch.float32)
# 绘制硅砖块（中心区域）
center = 64 // 2
layer[center-brick_half_pix:center+brick_half_pix,
      center-brick_half_pix:center+brick_half_pix] = 1.0
# 绘制内部空气孔（挖去中心区域）
layer[center-hole_half_pix:center+hole_half_pix,
      center-hole_half_pix:center+hole_half_pix] = 0.0

# ===================== 4. 构建RCWA仿真物理参数字典=====================
phy_kwargs = {
    "periodicity": P,    # 晶胞周期
    "h": H,              # 结构层厚度（砖高）
    "lam": lam,          # 入射波长
    "tet": tet,          # 入射角
    "substrate": substrate,  # 衬底材料
    "structure": structure   # 结构材料
}

# ===================== 5. 调用RCWA仿真函数（极简配置，论文一致）=====================
print(f"开始RCWA仿真：波长{lam}nm | 周期{P}nm | 入射角{tet}°")
# rcwa_orders=7：与rcwa.py示例一致，兼顾精度与速度
sim_result = torcwa_simulation(
    phy_kwargs=phy_kwargs,
    layer=layer,
    rcwa_orders=7,
    device=device
)

# ===================== 6. 提取并打印核心结果（论文重点关注|t_pp|）=====================
# 仿真结果：t_matrix是2x2复矩阵 [[t_ss, t_sp], [t_ps, t_pp]]
t_matrix = sim_result["t_matrix"]
t_ss, t_sp = t_matrix[0,0], t_matrix[0,1]
t_ps, t_pp = t_matrix[1,0], t_matrix[1,1]

# 打印复数值+模值（论文中均用模值|t(α)|表示透射系数，145行）
print("\n===================== 仿真结果（Jones矩阵-0阶透射）=====================")
print(f"t_ss (s-s偏振): {t_ss:.4f} | 模值: {torch.abs(t_ss):.4f}")
print(f"t_sp (s-p偏振): {t_sp:.4f} | 模值: {torch.abs(t_sp):.4f}")
print(f"t_ps (p-s偏振): {t_ps:.4f} | 模值: {torch.abs(t_ps):.4f}")
print(f"t_pp (p-p偏振): {t_pp:.4f} | 模值: {torch.abs(t_pp):.4f}")
print("========================================================================")
print(f"论文核心关注通道：p-p偏振透射模值 = {torch.abs(t_pp):.4f}")