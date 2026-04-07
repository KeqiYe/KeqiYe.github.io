function [W, grad_W] = CubicSplineKernel(pos_i, pos_j, h, dim)
    % 输入: 目标点 pos_i, 邻居点 pos_j, 光滑长度 h, 维度 dim
    % 输出: 核函数标量值 W, 核函数空间梯度向量 grad_W (方向为 i-j)
    
    r_vec = pos_i - pos_j;
    r = norm(r_vec);
    q = r / h;
    
    % 维度归一化系数
    if dim == 1, alpha = 2.0 / (3.0 * h);
    elseif dim == 2, alpha = 10.0 / (7.0 * pi * h^2);
    elseif dim == 3, alpha = 1.0 / (pi * h^3); 
    end
    
    W = 0.0;
    dWdr = 0.0;
    grad_W = zeros(1, dim);
    
    % 紧支域外直接返回 0
    if q >= 2.0
        return;
    end
    
    % 分段计算核函数值和距离导数
    if q >= 0 && q < 1
        W = alpha * (1.0 - 1.5*q^2 + 0.75*q^3);
        dWdr = (alpha / h) * (-3.0*q + 2.25*q^2);
    elseif q >= 1 && q < 2
        W = alpha * 0.25 * (2.0 - q)^3;
        dWdr = (alpha / h) * (-0.75 * (2.0 - q)^2);
    end
    
    % 计算空间梯度 (链式法则, 避免除以 0)
    if r > 1e-10
        grad_W = dWdr * (r_vec / r);
    end
end