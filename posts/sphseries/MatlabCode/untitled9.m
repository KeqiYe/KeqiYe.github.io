%% ======================== 全局参数 ======================================
DIM           = 3;
dp            = 0.05;
domain        = [0, 1];
kappa         = 2.0;
h_factor      = 1.3;
KGC_DET_THRESHOLD = 0.01;

fprintf('=== Balsara Switch Verification (DIM = %d) ===\n\n', DIM);

%% ======================== 粒子初始化 ====================================
[pos, vel, N] = initParticles(DIM, dp, domain);

rho   = 1000 * ones(N, 1);
if DIM == 2
    mass = rho .* dp^2;
else
    mass = rho .* dp^3;
end
h_arr = h_factor * dp * ones(N, 1);
cs    = 1.0 * ones(N, 1);    % ← 改为 1.0，使 ε 与测试梯度量级匹配

fprintf('粒子数 N = %d\n', N);
fprintf('h = %.4f,  ε = %.2e\n\n', h_arr(1), 1e-4 * cs(1) / h_arr(1));

%% ======================== 邻居搜索 ======================================
[neighbors, numNei] = findNeighbors(pos, h_arr, kappa, N, DIM);

%% ======================== 四个测试 ======================================
testNames = {'匀速平动', '均匀压缩', '刚体旋转', '纯剪切'};
expected_f = [0, 1, 0, 0];

for testID = 1:4
    vel = setVelocityField(testID, pos, DIM, N);
    [div_exact, curl_mag_exact] = analyticDivCurl(testID, DIM);
    
    L_all = computeKGC(pos, mass, rho, h_arr, neighbors, numNei, ...
                       N, DIM, kappa, KGC_DET_THRESHOLD);
    [balsara, div_v, curl_mag] = computeBalsara(pos, vel, mass, rho, ...
                                  h_arr, cs, neighbors, numNei, ...
                                  L_all, N, DIM, kappa);
    
    interior = getInteriorMask(pos, domain, h_arr, kappa, DIM);
    
    % ---- 额外诊断：打印各项量级 ----
    eps_val = 1e-4 * cs(1) / h_arr(1);
    abs_div_mean  = mean(abs(div_v(interior)));
    abs_curl_mean = mean(curl_mag(interior));
    
    fprintf('──────────────────────────────────────────\n');
    fprintf('测试 %d: %s\n', testID, testNames{testID});
    fprintf('──────────────────────────────────────────\n');
    fprintf('  理论散度       = %+.6f\n', div_exact);
    fprintf('  理论旋度模     = %+.6f\n', curl_mag_exact);
    fprintf('  期望 Balsara f ≈ %.2f\n', expected_f(testID));
    fprintf('  ---- 量级对比 ----\n');
    fprintf('  |div|  = %.6f\n', abs_div_mean);
    fprintf('  |curl| = %.6f\n', abs_curl_mean);
    fprintf('  ε      = %.6f\n', eps_val);
    fprintf('  |div|/ε = %.2f  (>>1 才能使 f→1)\n', abs_div_mean / eps_val);
    fprintf('  ---- 内部粒子统计 (N_int = %d) ----\n', sum(interior));
    fprintf('  SPH 散度  : mean = %+.6f,  std = %.2e\n', ...
            mean(div_v(interior)), std(div_v(interior)));
    fprintf('  SPH |旋度|: mean = %+.6f,  std = %.2e\n', ...
            mean(curl_mag(interior)), std(curl_mag(interior)));
    fprintf('  Balsara f : mean = %.6f,  std = %.2e,  [min, max] = [%.4f, %.4f]\n', ...
            mean(balsara(interior)), std(balsara(interior)), ...
            min(balsara(interior)), max(balsara(interior)));
    
    f_mean = mean(balsara(interior));
    tol = 0.15;
    if abs(f_mean - expected_f(testID)) < tol
        fprintf('  结果: ✅ PASS\n\n');
    else
        fprintf('  结果: ❌ FAIL  (偏差 = %.4f > %.2f)\n\n', ...
                abs(f_mean - expected_f(testID)), tol);
    end
end

%% ======================== 可视化 ========================================
figure('Position', [100 100 1400 900]);
for testID = 1:4
    vel = setVelocityField(testID, pos, DIM, N);
    L_all = computeKGC(pos, mass, rho, h_arr, neighbors, numNei, ...
                       N, DIM, kappa, KGC_DET_THRESHOLD);
    [balsara, ~, ~] = computeBalsara(pos, vel, mass, rho, ...
                        h_arr, cs, neighbors, numNei, ...
                        L_all, N, DIM, kappa);
    
    subplot(2, 2, testID);
    if DIM == 2
        scatter(pos(:,1), pos(:,2), 15, balsara, 'filled');
    else
        scatter3(pos(:,1), pos(:,2), pos(:,3), 15, balsara, 'filled');
    end
    colorbar; caxis([0 1]);
    title(sprintf('Test %d: %s  (expected f≈%.0f)', ...
          testID, testNames{testID}, expected_f(testID)));
    xlabel('x'); ylabel('y');
    axis equal tight;
end
sgtitle('Balsara Switch Verification', 'FontSize', 14);

%% ========================================================================
%                         子 函 数 定 义
%% ========================================================================

function [pos, vel, N] = initParticles(DIM, dp, domain)
    coords1d = (domain(1) + dp/2) : dp : (domain(2) - dp/2);
    if DIM == 2
        [X, Y] = meshgrid(coords1d, coords1d);
        pos = [X(:), Y(:)];
    else
        [X, Y, Z] = meshgrid(coords1d, coords1d, coords1d);
        pos = [X(:), Y(:), Z(:)];
    end
    N   = size(pos, 1);
    vel = zeros(N, DIM);
end

function vel = setVelocityField(testID, pos, DIM, N)
    vel = zeros(N, DIM);
    center = 0.5;
    x = pos(:,1) - center;
    y = pos(:,2) - center;
    
    switch testID
        case 1
            vel(:,1) = 1.0;
            if DIM >= 2, vel(:,2) = 0.5; end
        case 2
            alpha = 0.1;
            vel(:,1) = -alpha * x;
            if DIM >= 2, vel(:,2) = -alpha * y; end
            if DIM == 3
                z = pos(:,3) - center;
                vel(:,3) = -alpha * z;
            end
        case 3
            omega = 1.0;
            vel(:,1) = -omega * y;
            vel(:,2) =  omega * x;
        case 4
            gamma_val = 1.0;
            vel(:,1) = gamma_val * y;
    end
end

function [div_exact, curl_mag_exact] = analyticDivCurl(testID, DIM)
    switch testID
        case 1
            div_exact = 0; curl_mag_exact = 0;
        case 2
            alpha = 0.1;
            div_exact = -alpha * DIM; curl_mag_exact = 0;
        case 3
            omega = 1.0;
            div_exact = 0; curl_mag_exact = 2 * omega;
        case 4
            gamma_val = 1.0;
            div_exact = 0; curl_mag_exact = abs(gamma_val);
    end
end

function [neighbors, numNei] = findNeighbors(pos, h_arr, kappa, N, DIM)
    maxNei    = 200;
    neighbors = zeros(N, maxNei, 'int32');
    numNei    = zeros(N, 1, 'int32');
    for i = 1:N
        cnt = 0;
        for j = 1:N
            if i == j, continue; end
            dr = pos(i,:) - pos(j,:);
            dist = sqrt(sum(dr.^2));
            hij = 0.5 * (h_arr(i) + h_arr(j));
            if dist < kappa * hij
                cnt = cnt + 1;
                if cnt <= maxNei
                    neighbors(i, cnt) = j;
                end
            end
        end
        numNei(i) = min(cnt, maxNei);
    end
end

function [W, GradW] = KernelandGradKernel(q, h, DIM)
    if DIM == 1
        sigma = 2.0 / (3.0 * h);
    elseif DIM == 2
        sigma = 10.0 / (7.0 * pi * h^2);
    else
        sigma = 1.0 / (pi * h^3);
    end
    if q < 1.0
        W     = sigma * (1.0 - 1.5*q^2 + 0.75*q^3);
        dWdq  = sigma * (-3.0*q + 2.25*q^2);
    elseif q < 2.0
        W     = sigma * 0.25 * (2.0 - q)^3;
        dWdq  = sigma * (-0.75) * (2.0 - q)^2;
    else
        W    = 0.0;
        dWdq = 0.0;
    end
    GradW = dWdq / h;
end

function L_all = computeKGC(pos, mass, rho, h_arr, neighbors, numNei, ...
                            N, DIM, kappa, KGC_DET_THRESHOLD)
    L_all = zeros(N, DIM, DIM);
    for i = 1:N
        M = zeros(DIM, DIM);
        hi = h_arr(i);
        for kk = 1:numNei(i)
            j = neighbors(i, kk);
            drV = pos(i,:) - pos(j,:);
            drSq = sum(drV.^2);
            if drSq < 1e-20, continue; end
            dr = sqrt(drSq);
            hj  = h_arr(j);
            hij = 0.5 * (hi + hj);
            q   = dr / hij;
            [~, GradW] = KernelandGradKernel(q, hij, DIM);
            scalar_grad = -GradW / dr;
            GradWvec = scalar_grad * drV;
            m_j_div_rho_j = mass(j) / rho(j);
            M = M + m_j_div_rho_j * (drV' * GradWvec);
        end
        detM = det(M);
        if detM < KGC_DET_THRESHOLD
            L_all(i,:,:) = eye(DIM);
        else
            L_all(i,:,:) = inv(M);
        end
    end
end

function [balsara, div_v_all, curl_mag_all] = computeBalsara( ...
            pos, vel, mass, rho, h_arr, cs, neighbors, numNei, ...
            L_all, N, DIM, kappa)
    div_v_all    = zeros(N, 1);
    if DIM == 3
        curl_v_all = zeros(N, 3);
    else
        curl_v_all = zeros(N, 1);
    end
    curl_mag_all = zeros(N, 1);
    balsara      = zeros(N, 1);
    for i = 1:N
        hi = h_arr(i);
        L_i = squeeze(L_all(i,:,:));
        div_v = 0.0;
        if DIM == 3
            curl_v = [0; 0; 0];
        else
            curl_v_z = 0.0;
        end
        for kk = 1:numNei(i)
            j = neighbors(i, kk);
            drV = pos(i,:) - pos(j,:);
            dvV = vel(i,:) - vel(j,:);
            drSq = sum(drV.^2);
            if drSq < 1e-20, continue; end
            dr = sqrt(drSq);
            hj  = h_arr(j);
            hij = 0.5 * (hi + hj);
            q   = dr / hij;
            [~, GradW] = KernelandGradKernel(q, hij, DIM);
            scalar_grad = -GradW / dr;
            GradW_orig = scalar_grad * drV;
            GradW_corr = (L_i * GradW_orig')';
            m_j_div_rho_j = mass(j) / rho(j);
            div_v = div_v + m_j_div_rho_j * dot(dvV, GradW_corr);
            if DIM == 3
                cx = dvV(2)*GradW_corr(3) - dvV(3)*GradW_corr(2);
                cy = dvV(3)*GradW_corr(1) - dvV(1)*GradW_corr(3);
                cz = dvV(1)*GradW_corr(2) - dvV(2)*GradW_corr(1);
                curl_v = curl_v + m_j_div_rho_j * [cx; cy; cz];
            else
                cz = dvV(1)*GradW_corr(2) - dvV(2)*GradW_corr(1);
                curl_v_z = curl_v_z + m_j_div_rho_j * cz;
            end
        end
        div_v_all(i) = div_v;
        if DIM == 3
            curl_v_all(i,:) = curl_v';
            curl_mag_all(i) = norm(curl_v);
        else
            curl_v_all(i) = curl_v_z;
            curl_mag_all(i) = abs(curl_v_z);
        end
        abs_div  = abs(div_v);
        abs_curl = curl_mag_all(i);
        eps_val  = 1e-4 * cs(i) / hi;
        balsara(i) = abs_div / (abs_div + abs_curl + eps_val);
    end
end

function mask = getInteriorMask(pos, domain, h_arr, kappa, DIM)
    margin = kappa * max(h_arr);
    mask = true(size(pos, 1), 1);
    for d = 1:DIM
        mask = mask & (pos(:,d) > domain(1) + margin) ...
                    & (pos(:,d) < domain(2) - margin);
    end
end