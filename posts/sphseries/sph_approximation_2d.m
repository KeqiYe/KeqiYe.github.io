% SPH_APPROXIMATION_2D.m
% 描述:
% 使用 'Cubic Spline' 或 'Quadratic' 核函数，在二维空间中近似
% 函数 f(x,y) = sin(x)cos(y) 及其梯度 ∇f，并使用颜色云图进行可视化。

clear; clc; close all;

%% 1. 参数设置
Lx = 2*pi; Ly = 2*pi;    % 定义域
Nx = 100; Ny = 100;        % 各方向粒子数 (增加一些以获得更平滑的图像)
N = Nx * Ny;             % 总粒子数
dx = Lx / Nx; dy = Ly / Ny; % 粒子间距
Vj = dx * dy;            % 2D中的粒子"体积"
h = 1.5 * dx;            % 光滑长度

% --- 可调参数 ---
% 选择 'cubic' 或 'quadratic'
kernel_choice = 'cubic'; 

% 创建粒子网格和真实函数/导数值
[X, Y] = meshgrid(linspace(dx/2, Lx-dx/2, Nx), linspace(dy/2, Ly-dy/2, Ny));
pos = [X(:), Y(:)]; % 将粒子位置存储为 N x 2 的矩阵

f_true = sin(pos(:,1)) .* cos(pos(:,2));
grad_f_true = [cos(pos(:,1)).*cos(pos(:,2)), -sin(pos(:,1)).*sin(pos(:,2))];

%% 2. 定义核函数及其梯度 (2D)
% (这部分代码与之前完全相同)
switch kernel_choice
    case 'cubic'
        alpha_d = 15 / (7*pi*h^2); % 2D 归一化系数
        kernel = @(R) alpha_d * ( (R>=0 & R<1) .* (2/3 - R.^2 + 0.5*R.^3) + ...
                                  (R>=1 & R<2) .* (1/6 * (2-R).^3) );
        grad_W_dR = @(R) alpha_d * ( (R>=0 & R<1) .* (-2*R + 1.5*R.^2) + ...
                                     (R>=1 & R<2) .* (-0.5 * (2-R).^2) );
        support_domain = 2*h;
        
    case 'quadratic'
        alpha_d = 2 / (pi*h^2); % 2D 归一化系数
        kernel = @(R) alpha_d * (R>=0 & R<=2) .* (3/16*R.^2 - 3/4*R + 3/4);
        grad_W_dR = @(R) alpha_d * (R>=0 & R<=2) .* (6/16*R - 3/4);
        support_domain = 2*h;
        
    otherwise
        error("未知的核函数类型。请选择 'cubic' 或 'quadratic'。");
end
grad_kernel = @(R, r_vec, r) grad_W_dR(R) .* (1/h) .* r_vec ./ (r + 1e-9);

%% 3. 执行 SPH 求和
% (这部分代码与之前完全相同)
f_approx = zeros(N, 1);
grad_f_approx = zeros(N, 2);

fprintf('开始SPH计算 (N=%d)... 这可能需要一些时间。\n', N);
tic;
for i = 1:N
    sum_f = 0;
    sum_grad_f = [0, 0];
    for j = 1:N
        r_vec = pos(i,:) - pos(j,:);
        r = norm(r_vec);
        if r < support_domain && r > 0 % r > 0 to avoid self-contribution issues in grad
            R = r / h;
            W_ij = kernel(R);
            sum_f = sum_f + f_true(j) * W_ij * Vj;
            grad_W_ij = grad_kernel(R, r_vec, r);
            sum_grad_f = sum_grad_f + (f_true(j) - f_true(i)) * grad_W_ij * Vj;
        end
    end
    % 函数近似需要包含粒子自身贡献
    f_approx(i) = sum_f + f_true(i) * kernel(0) * Vj;
    grad_f_approx(i,:) = sum_grad_f;
end
toc;


%% 4. 可视化 (修改为颜色云图)

figure('Name', ['2D SPH Approximation with ' kernel_choice ' kernel'], 'Position', [50, 50, 1400, 800]);
colormap('jet'); % 设置颜色映射

% --- 准备绘图数据 (将向量reshape回网格) ---
F_true_grid = reshape(f_true, Ny, Nx);
F_approx_grid = reshape(f_approx, Ny, Nx);
Error_f_grid = F_true_grid - F_approx_grid;

dFdx_true_grid = reshape(grad_f_true(:,1), Ny, Nx);
dFdx_approx_grid = reshape(grad_f_approx(:,1), Ny, Nx);
Error_dfdx_grid = dFdx_true_grid - dFdx_approx_grid;

dFdy_true_grid = reshape(grad_f_true(:,2), Ny, Nx);
dFdy_approx_grid = reshape(grad_f_approx(:,2), Ny, Nx);
Error_dfdy_grid = dFdy_true_grid - dFdy_approx_grid;


% --- 绘制函数值 ---
subplot(2, 3, 1);
pcolor(X, Y, F_true_grid);
shading interp;
colorbar;
axis equal tight;
title('真实函数 f');
xlabel('x'); ylabel('y');

subplot(2, 3, 2);
pcolor(X, Y, F_approx_grid);
shading interp;
colorbar;
axis equal tight;
title('SPH 近似 f');
xlabel('x'); ylabel('y');

subplot(2, 3, 3);
pcolor(X, Y, Error_f_grid);
shading interp;
colorbar;
axis equal tight;
% 让误差的颜色条对称，更直观
max_err = max(abs(Error_f_grid(:)));
caxis([-max_err, max_err]); 
title('函数近似误差');
xlabel('x'); ylabel('y');


% --- 绘制x方向导数 ---
subplot(2, 3, 4);
pcolor(X, Y, dFdx_true_grid);
shading interp;
colorbar;
axis equal tight;
title('真实导数 ∂f/∂x');
xlabel('x'); ylabel('y');

subplot(2, 3, 5);
pcolor(X, Y, dFdx_approx_grid);
shading interp;
colorbar;
axis equal tight;
title('SPH 近似 ∂f/∂x');
xlabel('x'); ylabel('y');

subplot(2, 3, 6);
pcolor(X, Y, Error_dfdx_grid);
shading interp;
colorbar;
axis equal tight;
max_err = max(abs(Error_dfdx_grid(:)));
caxis([-max_err, max_err]); 
title('∂f/∂x 近似误差');
xlabel('x'); ylabel('y');