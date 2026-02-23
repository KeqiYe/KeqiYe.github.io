function [W, dWdr] = kernel_dkernel(r, h, dim, type)
% KERNEL_DKERNEL - 计算SPH核函数(W)及其对距离r的导数(dWdr)。
% (版本: 仅标量输入)
%
% 语法: [W, dWdr] = kernel_dkernel(r, h, dim, type)
%
% 输入:
%   r    - 粒子间的距离 |x - x'| (必须是标量)
%   h    - 光滑长度 (必须是标量)
%   dim  - 问题的维度 (1, 2, 或 3)
%   type - 核函数类型 (整数):
%          1 (钟形核函数, Lucy)
%          2 (B样条核函数)
%          3 (二次核函数)
%
% 输出:
%   W    - 核函数的值 (标量)
%   dWdr - 核函数对距离r的导数, dW/dr (标量)
%

% 1. 初始化
% 计算相对距离 R (r和h均为标量)
R = r / h;

% 初始化输出值为 0
W = 0.0;
dWdr = 0.0;

% 2. 根据类型选择核函数 (type 为整数)
switch type
    case 1
        % (1) 钟形核函数 (Lucy Kernel)
        % 定义域: 0 <= R <= 1
        
        % 计算归一化系数 alpha_d
        switch dim
            case 1
                alpha_d = 5 / (4 * h);
            case 2
                alpha_d = 5 / (pi * h^2);
            case 3
                alpha_d = 105 / (16 * pi * h^3);
            otherwise
                error('维度 (dim) 必须是 1, 2, 或 3。');
        end
        
        % 检查是否在定义域内
        if R >= 0 && R <= 1
            % W = alpha_d * (1 + 3R) * (1 - R)^3
            W = alpha_d * (1 + 3 * R) * (1 - R)^3;
            
            % dW/dR = alpha_d * -12 * R * (1 - R)^2
            dW_dR = alpha_d * (-12 * R) * (1 - R)^2;
            
            % dW/dr = (dW/dR) * (dR/dr) = (dW/dR) * (1/h)
            dWdr = dW_dR * (1/h);
        end
        % R > 1 或 R < 0 时, W 和 dWdr 保持为 0

    case 2
        % (3) B 样条核函数
        % 定义域: 0 <= R < 1 和 1 <= R < 2
        
        % 计算归一化系数 alpha_d
        switch dim
            case 1
                alpha_d = 1 / h;
            case 2
                alpha_d = 15 / (7 * pi * h^2);
            case 3
                alpha_d = 3 / (2 * pi * h^3);
            otherwise
                error('维度 (dim) 必须是 1, 2, 或 3。');
        end
        
        % ----- 区域 1: 0 <= R < 1 -----
        if R >= 0 && R < 1
            % W = alpha_d * (2/3 - R^2 + 1/2 * R^3)
            W = alpha_d * (2/3 - R^2 + 1/2 * R^3);
            
            % dW/dR = alpha_d * (-2*R + 3/2 * R^2)
            dW_dR = alpha_d * (-2*R + (3/2) * R^2);
            
            % dW/dr = (dW/dR) * (1/h)
            dWdr = dW_dR * (1/h);
            
        % ----- 区域 2: 1 <= R < 2 -----
        elseif R >= 1 && R < 2
            % W = alpha_d * (1/6 * (2 - R)^3)
            W = alpha_d * (1/6 * (2 - R)^3);
            
            % dW/dR = alpha_d * (-1/2 * (2 - R)^2)
            dW_dR = alpha_d * (-1/2 * (2 - R)^2);
            
            % dW/dr = (dW/dR) * (1/h)
            dWdr = dW_dR * (1/h);
        end
        % R >= 2 或 R < 0 时, W 和 dWdr 保持为 0

    case 3
        % (5) 二次核函数
        % 定义域: 0 <= R <= 2
        
        % 计算归一化系数 alpha_d
        switch dim
            case 1
                alpha_d = 1 / h;
            case 2
                alpha_d = 2 / (pi * h^2);
            case 3
                alpha_d = 5 / (4 * pi * h^3);
            otherwise
                error('维度 (dim) 必须是 1, 2, 或 3。');
        end
        
        % 检查是否在定义域内 (0 <= R <= 2)
        if R >= 0 && R <= 2
            % W = alpha_d * (3/16 * R^2 - 3/4 * R + 3/4)
            W = alpha_d * ( (3/16) * R^2 - (3/4) * R + 3/4 );
            
            % dW/dR = alpha_d * (3/8 * R - 3/4)
            dW_dR = alpha_d * ( (3/8) * R - 3/4 );
            
            % dW/dr = (dW/dR) * (1/h)
            dWdr = dW_dR * (1/h);
        end
        % R > 2 或 R < 0 时, W 和 dWdr 保持为 0
        
    otherwise
        error("无效的核函数类型 (type)。请输入 1, 2, 或 3。");
end

end