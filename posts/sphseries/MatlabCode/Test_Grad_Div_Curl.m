% =========================================================================
% 测试脚本 B: SPH 梯度、散度与旋度算子测试 (3D 空间)
% =========================================================================
clear; clc;

dim = 3;
dx = 0.15; h = 1.5 * dx; V_j = dx^dim; rho_0 = 1.0; m_j = rho_0 * V_j;

% 建立 3D 粒子域 
[X, Y, Z] = ndgrid(-0.6:dx:0.6, -0.6:dx:0.6, -0.6:dx:0.6);
particles = [X(:), Y(:), Z(:)];
num_particles = size(particles, 1);

% 挑选测试点: 中心点 和 边界点
test_indices = [floor(num_particles/2)+1, 1];
labels = {'中心内部点', '边缘角落点'};

fprintf('=== SPH 梯度、散度、旋度测试 (%dD) ===\n', dim);

for idx = 1:length(test_indices)
    target_idx = test_indices(idx);
    pos_i = particles(target_idx, :);
    exact_i = GetTestFunctions(pos_i, dim);
    
    % 初始化算子累加器
    approx_grad_f = zeros(1, 3);
    approx_div_v  = 0.0;
    approx_curl_v = zeros(1, 3);
    
    % 遍历计算
    for j = 1:num_particles
        pos_j = particles(j, :);
        [W_ij, grad_W_ij] = CubicSplineKernel(pos_i, pos_j, h, dim);
        
        if norm(grad_W_ij) > 0 % 如果导数不为0 (排除了自身和域外粒子)
            exact_j = GetTestFunctions(pos_j, dim);
            
            % 1. 梯度 (Grad): <nabla f> = sum (m/rho) * (f_j - f_i) * grad_W
            approx_grad_f = approx_grad_f + (m_j / rho_0) * (exact_j.f - exact_i.f) * grad_W_ij;
            
            % 2. 散度 (Div): <nabla . v> = sum (m/rho) * (v_j - v_i) dot grad_W
            approx_div_v = approx_div_v + (m_j / rho_0) * dot((exact_j.v - exact_i.v), grad_W_ij);
            
            % 3. 旋度 (Curl): <nabla x v> = sum (m/rho) * (v_j - v_i) cross grad_W
            % 由于前面 3D 定义 v 有3个分量，grad_W_ij 有3个分量，可以直接使用 cross
            approx_curl_v = approx_curl_v + (m_j / rho_0) * cross((exact_j.v - exact_i.v), grad_W_ij);
        end
    end
    
    % 打印结果对比
    fprintf('\n测试位置: %s, 坐标: [%.2f, %.2f, %.2f]\n', labels{idx}, pos_i(1), pos_i(2), pos_i(3));
    
    fprintf('  【梯度 Grad f】 (向量)\n');
    fprintf('    精确: [%7.4f, %7.4f, %7.4f]\n', exact_i.grad_f);
    fprintf('    近似: [%7.4f, %7.4f, %7.4f]\n', approx_grad_f);
    
    fprintf('  【散度 Div v】 (标量)\n');
    fprintf('    精确: %7.4f\n', exact_i.div_v);
    fprintf('    近似: %7.4f\n', approx_div_v);
    
    fprintf('  【旋度 Curl v】 (向量)\n');
    fprintf('    精确: [%7.4f, %7.4f, %7.4f]\n', exact_i.curl_v);
    fprintf('    近似: [%7.4f, %7.4f, %7.4f]\n', approx_curl_v);
end