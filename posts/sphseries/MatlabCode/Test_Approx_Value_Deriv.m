% =========================================================================
% 测试脚本 A: 标量函数值近似与单方向偏导数近似测试
% =========================================================================
clear; clc;

dim = 2;
dx = 0.1; h = 1.5 * dx; V_j = dx^dim; rho_0 = 1000.0; m_j = rho_0 * V_j;

% 建立 2D 粒子域 [-1, 1]
[X, Y] = ndgrid(-1:dx:1, -1:dx:1);
particles = [X(:), Y(:)];
num_particles = size(particles, 1);

% 挑选两个测试点: 中心点(内部) 和 右下角点(边界)
test_indices = [floor(num_particles/2)+1, num_particles];
labels = {'中心内部点', '边缘角落点'};

fprintf('=== SPH 函数重构与偏导数测试 (%dD) ===\n', dim);

for idx = 1:length(test_indices)
    target_idx = test_indices(idx);
    pos_i = particles(target_idx, :);
    
    % 获取目标粒子 i 的精确解析解
    exact_i = GetTestFunctions(pos_i, dim);
    
    % 初始化 SPH 近似累加器
    approx_f = 0.0;
    approx_dfdx = 0.0;
    
    % 遍历所有粒子 j 进行求和
    for j = 1:num_particles
        pos_j = particles(j, :);
        [W_ij, grad_W_ij] = CubicSplineKernel(pos_i, pos_j, h, dim);
        
        if W_ij > 0 % 在支持域内
            exact_j = GetTestFunctions(pos_j, dim);
            
            % 1. 函数值近似: <f_i> = sum (m/rho) * f_j * W_ij
            approx_f = approx_f + (m_j / rho_0) * exact_j.f * W_ij;
            
            % 2. 偏导数近似 (x方向): <df/dx> = sum (m/rho) * (f_j - f_i) * dW_dx
            % 注意: grad_W_ij(1) 就是 dW/dx
            approx_dfdx = approx_dfdx + (m_j / rho_0) * (exact_j.f - exact_i.f) * grad_W_ij(1);
        end
    end
    
    % 打印对比结果
    fprintf('\n测试位置: %s, 坐标: [%.2f, %.2f]\n', labels{idx}, pos_i(1), pos_i(2));
    fprintf('  函数值 f(x,y):   精确 = %7.4f, SPH近似 = %7.4f (误差 = %.4f)\n', ...
        exact_i.f, approx_f, abs(exact_i.f - approx_f));
    fprintf('  偏导数 df/dx:    精确 = %7.4f, SPH近似 = %7.4f (误差 = %.4f)\n', ...
        exact_i.dfdx, approx_dfdx, abs(exact_i.dfdx - approx_dfdx));
end