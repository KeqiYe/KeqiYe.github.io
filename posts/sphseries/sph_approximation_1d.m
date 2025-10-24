% SPH_APPROXIMATION_1D.m
% 描述:
% 使用 'Cubic Spline' 或 'Quadratic' 核函数，在一维空间中近似
% 函数 f(x) = sin(x) 及其导数 f'(x) = cos(x)。

clear; clc; close all;

%% 1. 参数设置
L = 2 * pi;             % 定义域长度
N = 100;                % 粒子数量
dx = L / N;             % 粒子间距 (也是1D中的粒子"体积")
h = 2.0 * dx;           % 光滑长度 (通常是dx的倍数)

% --- 可调参数 ---
% 选择 'cubic' 或 'quadratic'
kernel_choice = 'cubic'; 

% 创建粒子和真实函数/导数值
x = linspace(dx/2, L-dx/2, N)'; % 粒子位置
f_true = sin(x);                % 真实函数值
df_true = cos(x);               % 真实导数值

%% 2. 定义核函数及其导数 (1D)
% R = r/h, r = |xi - xj|
% dW/dx = (dW/dR) * (dR/dr) * (dr/dx) = (f'(R)/alpha_d) * (1/h) * sign(xi-xj)
% 我们将核函数 W 和它的导数 dW/dx 都定义为匿名函数

switch kernel_choice
    case 'cubic'
        alpha_d = 1/h; % 1D 归一化系数
        kernel = @(R) alpha_d * ( (R>=0 & R<1) .* (2/3 - R.^2 + 0.5*R.^3) + ...
                                  (R>=1 & R<2) .* (1/6 * (2-R).^3) );
        grad_kernel = @(R, r_vec) alpha_d * ( (R>=0 & R<1) .* (-2*R + 1.5*R.^2) + ...
                                          (R>=1 & R<2) .* (-0.5 * (2-R).^2) ) .* (1/h) .* sign(r_vec);
        support_domain = 2*h;
        
    case 'quadratic'
        alpha_d = 1/h; % 1D 归一化系数
        kernel = @(R) alpha_d * (R>=0 & R<=2) .* (3/16*R.^2 - 3/4*R + 3/4);
        grad_kernel = @(R, r_vec) alpha_d * (R>=0 & R<=2) .* (6/16*R - 3/4) .* (1/h) .* sign(r_vec);
        support_domain = 2*h;
        
    otherwise
        error("未知的核函数类型。请选择 'cubic' 或 'quadratic'。");
end

%% 3. 执行 SPH 求和 (对所有粒子)
f_approx = zeros(N, 1);
df_approx = zeros(N, 1);

for i = 1:N
    % 对每个粒子i，计算其近似值
    sum_f = 0;
    sum_df = 0;
    
    for j = 1:N
        r_vec = x(i) - x(j);
        r = abs(r_vec);
        
        if r < support_domain
            R = r / h;
            
            % 函数值近似 (公式: A(i) = Σ A(j) * W_ij * V_j)
            W_ij = kernel(R);
            sum_f = sum_f + f_true(j) * W_ij * dx;
            
            % 导数值近似 (公式: ∇A(i) = Σ [A(j)-A(i)] * ∇W_ij * V_j)
            grad_W_ij = grad_kernel(R, r_vec);
            sum_df = sum_df + (f_true(j) - f_true(i)) * grad_W_ij * dx;
        end
    end
    f_approx(i) = sum_f;
    df_approx(i) = sum_df;
end

%% 4. 计算误差
err_f = f_true - f_approx;
mae_f = mean(abs(err_f)); % 平均绝对误差
err_df = df_true - df_approx;
mae_df = mean(abs(err_df));

%% 5. 可视化
figure('Name', ['1D SPH Approximation with ' kernel_choice ' kernel'], 'Position', [50, 50, 1200, 800]);

% --- 子图1: 函数近似 ---
subplot(2, 2, 1);
hold on; grid on; box on;
plot(x, f_true, 'b-', 'LineWidth', 2, 'DisplayName', '真实函数 sin(x)');
plot(x, f_approx, 'r--', 'LineWidth', 2, 'DisplayName', 'SPH 近似');
title('函数近似');
xlabel('x'); ylabel('函数值');
legend('show', 'Location', 'northeast');

% --- 子图2: 函数近似误差 ---
subplot(2, 2, 2);
plot(x, err_f, 'g-', 'LineWidth', 1.5);
grid on; box on;
title(sprintf('函数近似误差 (MAE: %.4f)', mae_f));
xlabel('x'); ylabel('误差 (真实值 - 近似值)');

% --- 子图3: 导数近似 ---
subplot(2, 2, 3);
hold on; grid on; box on;
plot(x, df_true, 'b-', 'LineWidth', 2, 'DisplayName', '真实导数 cos(x)');
plot(x, df_approx, 'r--', 'LineWidth', 2, 'DisplayName', 'SPH 近似');
title('导数近似');
xlabel('x'); ylabel('导数值');
legend('show', 'Location', 'northeast');

% --- 子图4: 导数近似误差 ---
subplot(2, 2, 4);
plot(x, err_df, 'g-', 'LineWidth', 1.5);
grid on; box on;
title(sprintf('导数近似误差 (MAE: %.4f)', mae_df));
xlabel('x'); ylabel('误差 (真实值 - 近似值)');