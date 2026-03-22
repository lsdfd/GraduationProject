clear; clc; close all;

%% 参数设置
lambda = 1250e-9;          % 工作波长 [m]
k0 = 2 * pi / lambda;      % 波数
NA = 0.4;                 % 数值孔径
k_max = k0 * NA;           % 最大空间频率

% 图像参数
Lx = 502e-6;               % 图像宽度 [m]
Ly = 502e-6;               % 图像高度 [m]
Nx = 502;                  % 像素数（x方向）
Ny = 502;                  % 像素数（y方向）
dx = Lx / Nx;              % 空间采样间隔 [m]
dy = Ly / Ny;

% 输入偏振（可修改）
polarization = 'LCP';        % 选项：'x', 'y', 'x45'（45度）, 'x-45'（-45度）
switch polarization
    case 'x'
        e_in = [1; 0];     % x偏振
    case 'y'
        e_in = [0; 1];     % y偏振
    case 'x45'
        e_in = [1; 1]/sqrt(2);  % 45度线偏振
    case 'x-45'
        e_in = [1; -1]/sqrt(2); % -45度线偏振
    case 'RCP'
        e_in = [1; -1i] / sqrt(2);     % 右旋圆偏振
    case 'LCP'
        e_in = [1; 1i] / sqrt(2);      % 左旋圆偏振
end

%% 生成"正方形"二值图像 - 实心正方形，强度均为1
% 创建一个全黑的画布（黑色背景）
I_in = zeros(Ny, Nx);  % 直接使用double类型

% 正方形大小（边长为图像宽度的1/4）
square_size = min(Nx, Ny) / 16;

% 计算正方形的位置（居中）
center_x = Nx / 2;
center_y = Ny / 2;
half_size = square_size / 2;

% 定义正方形的边界
x_start = round(center_x - half_size);
x_end = round(center_x + half_size);
y_start = round(center_y - half_size);
y_end = round(center_y + half_size);

% 确保索引在范围内
x_start = max(1, x_start);
x_end = min(Nx, x_end);
y_start = max(1, y_start);
y_end = min(Ny, y_end);

% 绘制实心正方形（内部强度均为1）
I_in(y_start:y_end, x_start:x_end) = 1;

% 确保是二值图像（0或1）
I_in = double(I_in > 0.5);

% 显示原始图像
figure('Position', [100 100 800 400]);
subplot(1,2,1);
imagesc(I_in); colormap('gray'); axis image; title('');
xlabel('x (pixel)'); ylabel('y (pixel)');
xlim([150 350]);
ylim([150 350]);
set(gca, 'FontSize', 12);
colorbar;

% 输入标量场（振幅）
f_in = sqrt(I_in);  % 假设振幅与强度平方根成正比（二值图像即0或1）

% 波矢域网格（kx, ky）
kx = linspace(-k0, k0, Nx);
ky = linspace(-k0, k0, Ny);
[KX, KY] = meshgrid(kx, ky);

% 极坐标表示
k_rho = sqrt(KX.^2 + KY.^2);
theta = zeros(size(k_rho));
valid_idx = k_rho <= k0*NA;  % 只考虑传播波（忽略倏逝波）
theta(valid_idx) = asin(k_rho(valid_idx) / k0);
phi = atan2(KY, KX);

% 限制在 NA 内（可选，传递函数通常已在外部数据中体现）
%NA_mask = double(k_rho <= k_max);  % 1 在 NA 内，0 在 NA 外

%% 导入外部传递函数（如果提供）
use_external_t = true;  % 设为 true 并指定文件路径以导入外部数据
t_ss = ones(size(KX));   % 默认值（全通）
t_pp = ones(size(KX));
t_sp = zeros(size(KX));
t_ps = zeros(size(KX));

if use_external_t
    % 示例：从 Excel 文件读取，假设文件有两张表，分别为 t_ss 和 t_pp
    % 数据格式应为三列：kx, ky, amplitude
    % 注意：这里仅示意，实际使用时需根据文件格式调整
    t_ss = xlsread('G:\SZLab\Task\Si_based differentiator\proof\2.TOF\tss.xlsx');
    t_pp = xlsread('G:\SZLab\Task\Si_based differentiator\proof\2.TOF\tpp.xlsx');
    % 插值到当前网格
    % t_ss = griddata(data_ss(:,1), data_ss(:,2), data_ss(:,3), KX, KY, 'linear');
    % t_pp = griddata(data_pp(:,1), data_pp(:,2), data_pp(:,3), KX, KY, 'linear');
    % 将 NaN 替换为 0
    % t_ss(isnan(t_ss)) = 0;
    % t_pp(isnan(t_pp)) = 0;
    
    % 如果没有外部文件，使用理想拉普拉斯传递函数作为示例
    %t_ideal = (k_rho.^2) / (k_max^2);  % 归一化，使得在 k_rho = k_max 时为 1
    %t_ideal(k_rho > k_max) = 0;        % NA 外截止
    %t_ss = t_ideal;
    %t_pp = t_ideal;
else
    % 使用理想拉普拉斯传递函数（偏振无关）
    t_ideal = (k_rho.^2) / (k_max^2);  % 归一化，使得在 k_rho = k_max 时为 1
    t_ideal(k_rho > k_max) = 0;        % NA 外截止
    t_ss = t_ideal;
    t_pp = t_ideal;
end

% 确保传递函数在 NA 外为零
%t_ss = t_ss .* NA_mask;
%t_pp = t_pp .* NA_mask;

%% 角谱分解（输入）
% 标量场的傅里叶变换（零频居中）
F_in = fftshift(fft2(fftshift(f_in)));

% 计算每个频率点的 s 和 p 分量
E_p_in = (e_in(1) * cos(phi) + e_in(2) * sin(phi)) .* F_in;
E_s_in = cos(theta) .* (e_in(2) * cos(phi) - e_in(1) * sin(phi)) .* F_in;

%% 超表面滤波（输出角谱）
E_s_out = t_ss .* E_s_in + t_sp .* E_p_in;
E_p_out = t_ps .* E_s_in + t_pp .* E_p_in;

%% 转换回 xy 偏振基矢
% 注意：当 cos(theta) = 0 时需避免除零（此处因 NA=0.34，cosθ>0.94，安全）
cos_theta = cos(theta);
cos_theta(cos_theta == 0) = eps;  % 避免除零

E_x_out = cos(phi) .* E_p_out - (sin(phi) ./ cos_theta) .* E_s_out;
E_y_out = sin(phi) .* E_p_out + (cos(phi) ./ cos_theta) .* E_s_out;

%% 逆傅里叶变换得到空间域电场
E_x_space = ifftshift(ifft2(ifftshift(E_x_out)));
E_y_space = ifftshift(ifft2(ifftshift(E_y_out)));

% 输出强度
I_out = abs(E_x_space).^2 + abs(E_y_space).^2;

%% 绘图
subplot(1,2,2);
imagesc(I_out); colormap('parula'); axis image; 
%title(['输出图像 (偏振: ', polarization, ')']);
xlabel('x (pixel)'); ylabel('y (pixel)');
caxis([0 0.2]);
set(gca, 'FontSize', 12);
xlim([150 350]);
ylim([150 350]);
colorbar;

% 调整整体标题
%sgtitle('偏振无关二阶微分器边缘检测 - 正方形图案居中');

%% 可选：显示传递函数（可视化）
figure('Position', [100 100 1200 400]);
subplot(1,3,1);
imagesc(kx, ky, abs(t_ss)); axis image; colorbar;
xlabel('k_x'); ylabel('k_y'); title('|t_{ss}|');
set(gca, 'YDir', 'normal');

subplot(1,3,2);
imagesc(kx, ky, abs(t_pp)); axis image; colorbar;
xlabel('k_x'); ylabel('k_y'); title('|t_{pp}|');
set(gca, 'YDir', 'normal');

subplot(1,3,3);
plot(kx(Nx/2+1:end), abs(t_ss(Ny/2, Nx/2+1:end)), 'b-', 'LineWidth', 1.5); hold on;
plot(kx(Nx/2+1:end), abs(t_pp(Ny/2, Nx/2+1:end)), 'r--', 'LineWidth', 1.5);
xlabel('k_x'); ylabel('幅度'); legend('t_{ss}', 't_{pp}');
xlim([0, k_max*1.2]); grid on;
title('沿 k_x 轴的传递函数');