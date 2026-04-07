function exact = GetTestFunctions(pos, dim)
    % 输入: pos 为空间坐标 [x, y, z], dim 为空间维度
    % 输出: 包含函数值及其各阶导数解析解的结构体 exact
    
    x = pos(1);
    y = 0; z = 0;
    if dim >= 2, y = pos(2); end
    if dim == 3, z = pos(3); end
    
    exact = struct();
    
    if dim == 1
        % 1D 标量场与向量场
        exact.f = 2*x^2 - 3*x + 1;
        exact.v = x^2 + 2*x;
        
        % 1D 解析导数
        exact.dfdx = 4*x - 3;
        exact.grad_f = 4*x - 3;
        exact.div_v = 2*x + 2;
        exact.curl_v =0; % 1D 无旋度
        
    elseif dim == 2
        % % 2D 标量场与向量场
        % exact.f = 2*x^2 - y^2 - 3*x + 4*y;
        % exact.v = [x^2 + 2*y - 3*x,  y^2 - 3*x + 2*y];
        % 
        % % 2D 解析导数
        % exact.dfdx = 4*x - 3;
        % exact.grad_f = [4*x - 3, -2*y + 4];
        % exact.div_v = (2*x - 3) + (2*y + 2);
        % % 2D 旋度表现为标量 (Z方向)
        % exact.curl_v = [0, 0, (-3) - (2)]; 
        % 2D 线性场: f = 3x - 4y + 5
        % v = [2x + y, 3y - x]
        exact.f = 3*x - 4*y + 5;
        exact.v = [2*x + y, 3*y - x];
        
        exact.dfdx = 3;
        exact.grad_f = [3, -4];
        exact.div_v = 2 + 3; % dvx/dx + dvy/dy = 5
        exact.curl_v = [0, 0, (-1) - (1)]; % dvy/dx - dvx/dy = -2
        
    elseif dim == 3
        % 3D 标量场与向量场
        exact.f = 2*x^2 - y^2 + 3*z^2 - 3*x + 4*y - 2*z;
        exact.v = [x^2 + 2*y - 3*x,  y^2 - 3*z + 2*y,  -z^2 + x - z];
        
        % 3D 解析导数
        exact.dfdx = 4*x - 3;
        exact.grad_f = [4*x - 3, -2*y + 4, 6*z - 2];
        exact.div_v = (2*x - 3) + (2*y + 2) + (-2*z - 1);
        % 3D 旋度公式: [dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy]
        exact.curl_v = [0 - (-3), 0 - 1, (-3) - 2]; 
    end
end