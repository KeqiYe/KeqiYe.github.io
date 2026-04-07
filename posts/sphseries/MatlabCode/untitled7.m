clear; clc; close all;

%% 1. 参数设置
DIM = 3;            % 改为 1, 2, 或 3 进行测试
N_side = 20;        % 每边粒子数
h = 1.2;            % 光滑长度
dx = 1.0;

% 根据维度定义线性场梯度 A (Target Gradient)
if DIM == 1
    A = 1.5;        % 1D: dv/dx = 1.5
elseif DIM == 2
    A = [1.2, -0.5; 
         0.3,  0.8]; 
else
    A = [1.2, -0.5, 0.1; 
         0.3,  0.8, -0.2; 
         0.1,  0.2, 0.5];
end

%% 2. 生成粒子分布
if DIM == 1
    pos = (1:N_side)';
elseif DIM == 2
    [X, Y] = meshgrid(1:N_side, 1:N_side);
    pos = [X(:), Y(:)];
elseif DIM == 3
    [X, Y, Z] = meshgrid(1:N_side, 1:N_side, 1:N_side);
    pos = [X(:), Y(:), Z(:)];
end

numParticles = size(pos, 1);
% 添加随机扰动 (0.1倍间距)，模拟非均匀分布
pos = pos + 0*(rand(size(pos)) - 0.5) * 0.2; 

% 初始状态
densities = ones(numParticles, 1);
masses = ones(numParticles, 1) * (dx^DIM);

% 核心修正点：确保矩阵乘法维度正确 (N x DIM) * (DIM x DIM)' = (N x DIM)
velocities = pos * A'; 

%% 3. 计算 KGC 和速度梯度
computed_grad_v = zeros(numParticles, DIM, DIM);

% 选取中间粒子作为观测点
test_idx = round(numParticles / 2);

for i = 1:numParticles
    posi = pos(i, :);
    veli = velocities(i, :);
    M = zeros(DIM, DIM);
    
    % --- 循环 1: 计算 M 矩阵 ---
    for j = 1:numParticles
        drV = posi - pos(j, :); % 向量 r_i - r_j
        drSq = sum(drV.^2);
        dr = sqrt(drSq);
        
        if dr > 1e-12 && dr < 2*h
            [~, GW_mag] = KernelandGradKernel_Matlab(dr/h, h, DIM);
            scalar_grad = -GW_mag / dr; % 对应 C++ 中的 -GradW / dr
            gradW = scalar_grad * drV;  % \nabla W
            
            % M = \sum (m/rho) * (ri-rj) \otimes \nabla W
            M = M + (masses(j)/densities(j)) * (drV' * gradW);
        end
    end
    
    % 求逆得到 L 矩阵
    if DIM == 1
        if abs(M) > 1e-6, L = 1/M; else, L = 1; end
        % L = 1;
    else
        if det(M) > 1e-6, L = inv(M); else, L = eye(DIM); end
        % L = eye(DIM);
    end
    
    % --- 循环 2: 计算修正后的速度梯度 ---
    grad_v_node = zeros(DIM, DIM);
    for j = 1:numParticles
        drV = posi - pos(j, :);
        dvV = veli - velocities(j, :); % v_i - v_j
        dr = sqrt(sum(drV.^2));
        
        if dr > 1e-12 && dr < 2*h
            [~, GW_mag] = KernelandGradKernel_Matlab(dr/h, h, DIM);
            scalar_grad = -GW_mag / dr;
            gradW = scalar_grad * drV;
            
            % 修正梯度: \tilde{\nabla}W = L * \nabla W
            corrected_gradW = (L * gradW')';
            
            % \nabla v = \sum (m/rho) * (vi-vj) \otimes \tilde{\nabla}W
            grad_v_node = grad_v_node + (masses(j)/densities(j)) * (dvV' * corrected_gradW);
        end
    end
    computed_grad_v(i,:,:) = grad_v_node;
end

%% 4. 结果展示
fprintf('--- 维度 DIM = %d 测试 ---\n', DIM);
res_A = squeeze(computed_grad_v(test_idx, :, :));
fprintf('预设 A:\n'); disp(A);
fprintf('计算结果:\n'); disp(res_A);
fprintf('最大绝对误差: %e\n', max(abs(res_A(:) - A(:))));

%% 5. 核函数
function [w, gw] = KernelandGradKernel_Matlab(q, h, dim)
    if dim == 1,     alpha = 2/(3*h);
    elseif dim == 2, alpha = 10/(7*pi*h^2);
    elseif dim == 3, alpha = 1/(pi*h^3);
    end
    w = 0; gw = 0;
    if q >= 0 && q < 1
        w = 1 - 1.5*q^2 + 0.75*q^3;
        gw = -3*q + 2.25*q^2;
    elseif q >= 1 && q < 2
        w = 0.25 * (2-q)^3;
        gw = -0.75 * (2-q)^2;
    end
    w = w * alpha;
    gw = gw * alpha / h;
end