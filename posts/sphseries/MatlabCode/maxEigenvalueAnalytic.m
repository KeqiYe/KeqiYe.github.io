function lambda_max = maxEigenvalueAnalytic(A)
% A: 2x2 或 3x3 对称矩阵

DIM = size(A,1);

if DIM == 3
    % ===== 提取元素 =====
    s11 = A(1,1); s12 = A(1,2); s13 = A(1,3);
    s22 = A(2,2); s23 = A(2,3);
    s33 = A(3,3);

    % ===== 1. 判断是否接近对角 =====
    p1 = s12^2 + s13^2 + s23^2;

    if p1 < 1e-16
        lambda_max = max([s11, s22, s33]);
        return;
    end

    % ===== 2. 平均应力 =====
    q = (s11 + s22 + s33) / 3.0;

    % ===== 3. 偏应力 =====
    d11 = s11 - q;
    d22 = s22 - q;
    d33 = s33 - q;

    % ===== 4. p =====
    p2 = d11^2 + d22^2 + d33^2 + 2*p1;
    p = sqrt(p2 / 6.0);

    % ===== 5. det(B) * p^3 =====
    det_B_p3 = d11 * (d22 * d33 - s23^2) ...
             - s12 * (s12 * d33 - s23 * s13) ...
             + s13 * (s12 * s23 - d22 * s13);

    % ===== 6. r =====
    r = det_B_p3 / (2.0 * p^3);

    % 数值保护
    r = max(min(r, 1.0), -1.0);

    % ===== 7. 最大特征值 =====
    phi = acos(r) / 3.0;

    lambda_max = q + 2.0 * p * cos(phi);

elseif DIM == 2
    % ===== 2D 情况 =====
    s11 = A(1,1);
    s12 = A(1,2);
    s22 = A(2,2);

    trace = s11 + s22;
    detA = s11 * s22 - s12^2;

    delta = trace^2 - 4.0 * detA;
    delta = max(delta, 0.0);

    lambda_max = 0.5 * (trace + sqrt(delta));

elseif DIM == 1
    lambda_max = A(1);

else
    error('Only supports 1D/2D/3D');
end

end