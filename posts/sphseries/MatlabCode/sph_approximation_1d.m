% SPH_INTERPOLATION_1D.m
% 描述:
% 将函数 f(x) 离散到一组"数据粒子"上。
% 然后使用 SPH 方法，在另一组"评估点"上插值重构 f(x) 及其导数 f'(x)，
% 并与真实值进行比较。这明确体现了 SPH 的插值思想。

clear; clc; close all;

%% 1. 参数设置
L = 2 * pi;             % 定义域长度
N_particles = 40;       % 数据粒子的数量 (故意减少，以体现插值的效果)
dx = L / N_particles;   % 数据粒子间距 (也是粒子"体积")
h = 2.0 * dx;           % 光滑长度

% --- 可调参数 ---
kernel_choice = 'cubic'; % 选择 'lucy', 'cubic', 或 'quadratic'

%% 2. 设置核函数参数
switch lower(kernel_choice)
    case 'lucy'
        kernel_type = 1; kappa = 1.0;
    case 'cubic'
        kernel_type = 2; kappa = 2.0;
    case 'quadratic'
        kernel_type = 3; kappa = 2.0;
    otherwise
        error("无效的核函数选择。");
end

%% 3. 创建"数据粒子"和"评估点"

% --- 数据粒子 (我们已知信息的点) ---
x_particles = linspace(dx/2, L-dx/2, N_particles)';   % 数据粒子的位置
f_particles = sin(x_particles) + cos(x_particles);      % 数据粒子上的函数值 (已知)
% 注意：我们只使用粒子的函数值 f_particles，不使用其导数值来插值

% --- 评估点 (我们想要插值的未知点) ---
N_eval = 200;                                           % 创建更密集的评估网格
x_eval = linspace(0, L, N_eval)';                       % 评估点的位置
f_true_eval = sin(x_eval) + cos(x_eval);                % 评估点上的真实函数值 (用于计算误差)
df_true_eval = cos(x_eval) - sin(x_eval);              % 评估点上的真实导数值 (用于计算误差)


%% 4. 在评估点上执行 SPH 插值
f_approx_eval = zeros(N_eval, 1);
df_approx_eval = zeros(N_eval, 1);

% 遍历每一个"评估点" i
for i = 1:N_eval
    
    sum_f = 0;
    
    % 遍历所有"数据粒子" j 来计算对评估点 i 的贡献
    for j = 1:N_particles
        r_vec = x_eval(i) - x_particles(j);
        r = abs(r_vec);

        if r < kappa * h
            % 调用核函数
            [W_ij, ~] = kernel_dkernel(r, h, 1, kernel_type);

            % 函数值插值 (公式: A(i) = Σ A(j) * W_ij * V_j)
            sum_f = sum_f + f_particles(j) * W_ij * dx;
        end
    end
    f_approx_eval(i) = sum_f;

    sum_df = 0;
    for j = 1:N_particles
        r_vec = x_eval(i) - x_particles(j);
        r = abs(r_vec);

        if r < kappa * h
            % 调用核函数
            [~, dWdr_ij] = kernel_dkernel(r, h, 1, kernel_type);

            % 导数值插值 (公式: ∇A(i) = Σ [A(j)-A(i)] * ∇W_ij * V_j)
            dWdx_ij = dWdr_ij * sign(r_vec);
            sum_df = sum_df + (f_particles(j) - f_approx_eval(i)) * dWdx_ij * dx;
        end
    end
    df_approx_eval(i) = sum_df;
end

%% 5. 计算误差
err_f = f_true_eval - f_approx_eval;
mae_f = mean(abs(err_f));
err_df = df_true_eval - df_approx_eval;
mae_df = mean(abs(err_df));

%% 6. 可视化
figure('Name', ['1D SPH Interpolation with ' kernel_choice ' kernel'], 'Position', [50, 50, 1200, 800]);

% --- 子图1: 函数插值 ---
subplot(2, 2, 1);
hold on; grid on; box on;
plot(x_eval, f_true_eval, 'b-', 'LineWidth', 2, 'DisplayName', '真实函数');
plot(x_eval, f_approx_eval, 'r--', 'LineWidth', 2, 'DisplayName', 'SPH 插值');
scatter(x_particles, f_particles, 60, 'k', 'filled', 'DisplayName', '数据粒子');
title('函数插值');
xlabel('x'); ylabel('函数值');
legend('show', 'Location', 'best');

% --- 子图2: 函数插值误差 ---
subplot(2, 2, 2);
plot(x_eval, err_f, 'g-', 'LineWidth', 1.5);
grid on; box on;
title(sprintf('函数插值误差 (MAE: %.4f)', mae_f));
xlabel('x'); ylabel('误差');

% --- 子图3: 导数插值 ---
subplot(2, 2, 3);
hold on; grid on; box on;
plot(x_eval, df_true_eval, 'b-', 'LineWidth', 2, 'DisplayName', '真实导数');
plot(x_eval, df_approx_eval, 'r--', 'LineWidth', 2, 'DisplayName', 'SPH 插值');
title('导数插值');
xlabel('x'); ylabel('导数值');
legend('show', 'Location', 'best');

% --- 子图4: 导数插值误差 ---
subplot(2, 2, 4);
plot(x_eval, err_df, 'g-', 'LineWidth', 1.5);
grid on; box on;
title(sprintf('导数插值误差 (MAE: %.4f)', mae_df));
xlabel('x'); ylabel('误差');

