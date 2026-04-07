% =========================================================================
% SPH Tillotson P-alpha Newton Iteration Debugger
% 1:1 replication of the CUDA kernel logic
% =========================================================================
clear; clc; close all;
% %% 1. 输入 GDB 捕获的粒子状态 (Particle ID : 154903)
% rho = 1821.77880859375;
% e = 1668872.125;
% % 注意：GDB 输出中报错停在了 3.4477，我们假设这是上一时间步传进来的历史最小孔隙度
% alpha_old = 1.0013741254806519; 
rho = 675.4053955078125;
e = 3157.989013671875;
alpha_old = 3.4477;

%% 2. 材料参数配置
mat.rho0 = 2327.0;
mat.A = 26.7e9;
mat.B = 26.7e9;
mat.E0 = 487.0e6;
mat.Eiv = 4.72e6;
mat.Ecv = 18.2e6;
mat.a = 0.5;
mat.b = 1.5;
mat.alpha_eos = 5.0; 
mat.beta = 5.0;
mat.P_min = -100e9;

% 孔隙度参数
mat.alpha_0 = 3.448275862068965;
mat.P_e = 1.0e6;
mat.P_s = 2.13e8;

%% 3. 安全牛顿法主循环设置
max_iters = 50;
tol = 1e-12;

alpha_k = alpha_old; 
alpha_low = 1.0;          % 绝对安全区间下限
alpha_high = alpha_old;   % 绝对安全区间上限

history.alpha = zeros(max_iters, 1);
history.F_k = zeros(max_iters, 1);
history.P_k = zeros(max_iters, 1);
history.action = strings(max_iters, 1); % 记录每步用了什么方法

fprintf('=== 开始安全牛顿迭代 (Particle ID: 154903) ===\n');
fprintf('Iter |     alpha_k     |        F_k        |   区间 [low, high]   | 采取动作\n');
fprintf('--------------------------------------------------------------------------------------\n');

converged = false;

for iter = 1:max_iters
    % 1 & 2. 算状态方程和宏观压强
    rho_s = alpha_k * rho;
    [P_eos, ~, dpde, dpdrho] = calculateTillotson(rho_s, e, mat);
    P_k = P_eos / alpha_k;
    
    % 3. 压溃曲线及导数
    [alpha_curve, d_alpha_curve_dP] = evaluateCrushCurve(P_k, mat, alpha_old);
    
    % 4. 计算残差 F_k
    F_k = alpha_k - alpha_curve;
    
    % 记录状态
    history.alpha(iter) = alpha_k;
    history.P_k(iter) = P_k;
    history.F_k(iter) = F_k;
    
    % ==========================================
    % [核心逻辑 1]：利用当前信息，收紧安全区间
    % ==========================================
    if F_k > 0.0
        alpha_high = alpha_k;
    else
        alpha_low = alpha_k;
    end
    
    % 打印当前状态
    fprintf('%4d | %15.8e | %17.8e | [%17.8e, %17.8e] | ', iter, alpha_k, F_k, alpha_low, alpha_high);
    
    % 收敛条件 1
    if abs(F_k) < tol
        converged = true;
        fprintf('收敛！\n');
        break;
    end
    
    % 5. 算导数
    dP_dalpha = (dpdrho * rho) / alpha_k - P_k / alpha_k;
    dF_dalpha = 1.0 - d_alpha_curve_dP * dP_dalpha;
    
    % 6. 纯牛顿步试探
    alpha_new = alpha_k - F_k / dF_dalpha;
    
    % ==========================================
    % [核心逻辑 2]：安全网拦截机制
    % ==========================================
    if abs(dF_dalpha) <= 1e-14 || alpha_new < alpha_low || alpha_new >= alpha_high
        % 越界或奇异！没收牛顿指挥权，强行二分折半
        alpha_new = 0.5 * (alpha_low + alpha_high);
        fprintf('🚫 越界拦截 -> 二分折半\n');
        history.action(iter) = "Bisection";
    else
        % 在安全区间内，采纳牛顿步
        fprintf('⚡ 纯牛顿步\n');
        history.action(iter) = "Newton";
    end
    
    % 收敛条件 2
    if abs(alpha_new - alpha_k) < tol
        alpha_k = alpha_new;
        converged = true;
        fprintf('     (下一步变化极小，收敛)\n');
        break;
    end
    
    alpha_k = alpha_new;
end

history.alpha = history.alpha(1:iter);
history.F_k = history.F_k(1:iter);
history.P_k = history.P_k(1:iter);

% 计算最终跳出循环时的 P_k
rho_s_final = alpha_k * rho;
[P_eos_final, ~, ~, ~] = calculateTillotson(rho_s_final, e, mat);
P_final = P_eos_final / alpha_k;

% 追加终点数据
history.alpha = [history.alpha; alpha_k];
history.P_k = [history.P_k; P_final];
history.F_k = [history.F_k; 0]; % 终点误差视为 0

if ~converged
    fprintf('\n>> 失败: 未能在 %d 步内收敛。\n', max_iters);
else
    fprintf('\n>> 成功！总迭代次数: %d 步。\n', iter);
end

%% 4. 绘图对比
% figure('Position',, 'Name', 'Safe Newton Debugger');

% 误差收敛图
subplot(1, 2, 1);
plot(1:length(history.F_k)-1, abs(history.F_k(1:end-1)), '-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
set(gca, 'YScale', 'log'); 
grid on; xlabel('Iteration'); ylabel('|F(\alpha)|');
title('Residual Error Decay');

% 压力-孔隙度 横跳终结图
subplot(1, 2, 2);
P_plot = linspace(0, mat.P_s * 1.1, 500);
alpha_plot = zeros(size(P_plot));
for i=1:length(P_plot)
    [alpha_plot(i), ~] = evaluateCrushCurve(P_plot(i), mat, alpha_old);
end

% 1. 画理论压溃曲线 (黑线)
plot(P_plot/1e6, alpha_plot, 'k-', 'LineWidth', 2); hold on;

% 2. 画迭代路径 (蓝线黄点)
plot(history.P_k/1e6, history.alpha, 'b--o', 'LineWidth', 1.5, 'MarkerFaceColor', 'y');

% % 3. ⭐️ 特别高亮最终收敛点 (红色大五角星)
% plot(history.P_k(end)/1e6, history.alpha(end), 'p', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');

grid on; xlabel('Pressure (MPa)'); ylabel('\alpha');
title('Safe Phase Space Path');
legend('Crush Curve', 'Solver Steps', 'Final Converged Root', 'Location', 'best');


%% ======================= 局部函数定义 ====================================

function [alpha_curve, deriv] = evaluateCrushCurve(P, mat, alpha_old)
    if P <= mat.P_e
        alpha_curve = mat.alpha_0;
        deriv = 0.0;
    elseif P >= mat.P_s
        alpha_curve = 1.0;
        deriv = 0.0;
    else
        ratio = (mat.P_s - P) / (mat.P_s - mat.P_e);
        alpha_curve = 1.0 + (mat.alpha_0 - 1.0) * (ratio * ratio);
        deriv = -2.0 * (mat.alpha_0 - 1.0) * ratio / (mat.P_s - mat.P_e);
    end
    
    if alpha_curve > alpha_old
        alpha_curve = alpha_old;
        deriv = 0.0;
    end
end

function [P, soundSpeed, dpde, dpdrho] = calculateTillotson(rho, e, mat)
    rho0_inv = 1.0 / mat.rho0;
    eta = rho * rho0_inv;
    mu = eta - 1.0;
    
    eta_sq = eta * eta;
    eta_cube = eta_sq * eta;
    
    inner_inv = safe_inv(mat.E0 * eta_sq, 1e-10);
    ack = safe_inv(1.0 + e * inner_inv, 1e-10);
    
    phi = mat.b * ack;
    dphidu = -phi * ack * inner_inv;
    dphidrho = 2.0 * phi * ack * e * safe_inv(mat.rho0 * mat.E0 * eta_cube, 1e-10);
    
    chi = safe_inv(eta, 1e-10) - 1.0;
    dchidrho = -safe_inv(mat.rho0 * eta_sq, 1e-10);
    
    P = 0.0; dPdu = 0.0; dPdrho = 0.0;
    
    if mu >= 0.0
        P = (mat.a + phi) * rho * e + mat.A * mu + mat.B * mu * mu;
        dPdu = (mat.a + phi) * rho + rho * e * dphidu;
        dPdrho = (mat.a + phi) * e + rho * e * dphidrho + (mat.A + 2.0 * mat.B * mu) * rho0_inv;
    elseif e <= mat.Eiv
        P = (mat.a + phi) * rho * e + mat.A * mu;
        dPdu = (mat.a + phi) * rho + rho * e * dphidu;
        dPdrho = (mat.a + phi) * e + rho * e * dphidrho + mat.A * rho0_inv;
    elseif e >= mat.Ecv
        exp_alpha = exp(-mat.alpha_eos * chi * chi);
        exp_beta = exp(-mat.beta * chi);
        P = mat.a * rho * e + (phi * rho * e + mat.A * mu * exp_beta) * exp_alpha;
        dPdu = mat.a * rho + rho * (phi + e * dphidu) * exp_alpha;
        dPdrho = mat.a * e + ...
                 (-(phi * rho * e + mat.A * mu * exp_beta) * 2.0 * mat.alpha_eos * chi * dchidrho + ...
                  (phi + rho * dphidrho) * e + ...
                  mat.A * (rho0_inv - mu * mat.beta * dchidrho) * exp_beta) * exp_alpha;
    else
        P2 = (mat.a + phi) * rho * e + mat.A * mu;
        dP2du = (mat.a + phi) * rho + rho * e * dphidu;
        dP2drho = (mat.a + phi) * e + rho * e * dphidrho + mat.A * rho0_inv;
        
        exp_alpha = exp(-mat.alpha_eos * chi * chi);
        exp_beta = exp(-mat.beta * chi);
        
        P4 = mat.a * rho * e + (phi * rho * e + mat.A * mu * exp_beta) * exp_alpha;
        dP4du = mat.a * rho + rho * (phi + e * dphidu) * exp_alpha;
        dP4drho = mat.a * e + ...
                  (-(phi * rho * e + mat.A * mu * exp_beta) * 2.0 * mat.alpha_eos * chi * dchidrho + ...
                   (phi + rho * dphidrho) * e + ...
                   mat.A * (rho0_inv - mu * mat.beta * dchidrho) * exp_beta) * exp_alpha;
               
        denomInv = safe_inv(mat.Ecv - mat.Eiv, 1e-10);
        w = (e - mat.Eiv) * denomInv;
        
        P = P2 + (P4 - P2) * w;
        dPdu = dP2du + (P4 - P2) * denomInv + (dP4du - dP2du) * w;
        dPdrho = dP2drho + (dP4drho - dP2drho) * w;
    end
    
    P_lim = max(mat.P_min, P);
    if P ~= P_lim
        dPdu = 0.0;
        dPdrho = 0.0;
    end
    P = P_lim;
    
    c2 = dPdrho + dPdu * P * safe_inv(rho * rho, 1e-10);
    min_c2 = 1e-4 * mat.A * rho0_inv;
    c2 = max(c2, min_c2);
    
    soundSpeed = sqrt(max(0.0, c2));
    dpde = dPdu;
    dpdrho = dPdrho;
end

function out = safe_inv(val, tol)
    if abs(val) < tol
        if val >= 0
            out = 1.0 / tol;
        else
            out = -1.0 / tol;
        end
    else
        out = 1.0 / val;
    end
end