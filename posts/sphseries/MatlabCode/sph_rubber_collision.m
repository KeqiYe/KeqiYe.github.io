function sph_rubber_collision_grid()
    % --- 1. 物理与材料参数 ---
    DIM = 2;
    rho0 = 1000;        
    h0 = 0.12;          % 平滑长度
    spacing = 0.05;     % 初始粒子间距 (网格步长)
    m0 = rho0 * (spacing^2); % 质量 = 密度 * 占有面积
    
    mu = 1e6;           % 剪切模量
    K = 5e6;            % 体积模量 (适当增大以模拟不可压缩性)
    alpha_v = 1.0;      
    beta_v = 2.0;
    
    % --- 2. 规则网格生成粒子 ---
    [pos1, vel1] = create_ring_grid(spacing, 0.8, 1.0, [-1.2, 0], [2.0, 0]);
    [pos2, vel2] = create_ring_grid(spacing, 0.8, 1.0, [1.2, 0], [-2.0, 0]);
    
    pos = [pos1; pos2];
    vel = [vel1; vel2];
    N = size(pos, 1);
    fprintf('粒子总数: %d\n', N);
    
    % 初始状态打包
    initial_rho = ones(N, 1) * rho0;
    initial_u = ones(N, 1) * 100; 
    initial_S = zeros(N, 4);      
    Y0 = [pos(:); vel(:); initial_rho; initial_u; initial_S(:)];
    
% --- 3. 高精度 ODE 求解 ---
    tspan = [0 1.2];
    
    % 定义 OutputFcn 来显示进度和 dt
    options = odeset('RelTol', 1e-4, 'AbsTol', 1e-4, 'OutputFcn', @sph_monitor); 
    
    fprintf('开始模拟计算...\n');
    fprintf('Time(s)    Progress(%%)    dt(s)\n');
    fprintf('---------------------------------\n');
    
    tic;
    [T, Y] = ode45(@(t, y) sph_system(t, y, N, m0, h0, mu, K, alpha_v, beta_v, rho0), tspan, Y0, options);
    toc;
    
    % --- 4. 模拟结束后统一绘图 ---
    fprintf('计算完成，开始播放动画...\n');
    figure('Color', 'w');
    for k = 1:5:length(T) % 每隔5帧画一次
        current_Y = Y(k, :);
        p_idx = reshape(current_Y(1:N*2), N, 2);
        
        % 根据速度大小着色，增加视觉精度
        v_idx = reshape(current_Y(N*2+1:N*4), N, 2);
        v_mag = sqrt(sum(v_idx.^2, 2));
        
        scatter(p_idx(:,1), p_idx(:,2), 15, v_mag, 'filled');
        colormap(jet);
        axis([-2.5 2.5 -1.5 1.5]);
        axis equal;
        title(['SPH Rubber Collision - Time: ', num2str(T(k), '%.3f')]);
        drawnow;
    end
end

% --- 核心计算函数 (修正维度) ---
function dY = sph_system(t, Y, N, m0, h0, mu, K, alpha, beta, rho0)
    % 解包 (同前，确保维度正确)
    pos = reshape(Y(1:N*2), N, 2);
    vel = reshape(Y(N*2+1:N*4), N, 2);
    rho = Y(N*4+1:N*5);
    u   = Y(N*5+1:N*6);
    S   = reshape(Y(N*6+1:end), N, 4);
    
    P = K * ((rho / rho0).^7 - 1); 
    cs = sqrt(7 * K * (rho.^6 / rho0^7)); 
    
    % 预计算修正矩阵 L (KGC)
    L = zeros(N, 4);
    for i = 1:N
        M = zeros(2, 2);
        for j = 1:N
            r_ij = pos(i,:) - pos(j,:);
            dist_sq = sum(r_ij.^2);
            if dist_sq > (2*h0)^2 || dist_sq < 1e-12, continue; end
            dist = sqrt(dist_sq);
            [~, gradW_val] = cubic_spline_kernel(dist, h0);
            M = M + (m0 / rho(j)) * (r_ij' * (gradW_val * r_ij / dist));
        end
        L(i,:) = reshape(inv(M + eye(2)*1e-7), 1, 4);
    end
    
    % 计算 Balsara 开关
    balsara = zeros(N, 1);
    for i = 1:N
        div_v = 0; curl_v = 0;
        Li = reshape(L(i,:), 2, 2);
        for j = 1:N
            r_ij = pos(i,:) - pos(j,:);
            dist = norm(r_ij);
            if dist > 2*h0 || dist < 1e-12, continue; end
            v_ij = vel(i,:) - vel(j,:);
            [~, gradW_val] = cubic_spline_kernel(dist, h0);
            gradW_corr = Li * (gradW_val * r_ij' / dist);
            div_v = div_v + (m0/rho(j)) * dot(v_ij, gradW_corr);
            curl_v = curl_v + (m0/rho(j)) * (v_ij(1)*gradW_corr(2) - v_ij(2)*gradW_corr(1));
        end
        balsara(i) = abs(div_v) / (abs(div_v) + abs(curl_v) + 0.01*cs(i)/h0);
    end

    d_pos = vel;
    d_vel = zeros(N, 2);
    d_rho = zeros(N, 1);
    d_u = zeros(N, 1);
    d_S = zeros(N, 4);
    
    for i = 1:N
        Li = reshape(L(i,:), 2, 2);
        Si = reshape(S(i,:), 2, 2);
        gradV = zeros(2, 2);
        
        for j = 1:N
            if i == j, continue; end
            r_ij = pos(i,:) - pos(j,:);
            dist_sq = sum(r_ij.^2);
            if dist_sq > (2*h0)^2, continue; end
            dist = sqrt(dist_sq);
            
            v_ij = vel(i,:) - vel(j,:);
            [~, gradW_val] = cubic_spline_kernel(dist, h0);
            gradW_orig = (gradW_val * r_ij' / dist); 
            gradW_corr = Li * gradW_orig;
            
            d_rho(i) = d_rho(i) + m0 * dot(v_ij, gradW_corr);
            
            dot_rv = dot(r_ij, v_ij);
            PI_ij = 0;
            if dot_rv < 0
                mu_ij = (h0 * dot_rv) / (dist_sq + 0.01*h0^2);
                PI_ij = (-alpha * 0.5*(cs(i)+cs(j)) * mu_ij + beta * mu_ij^2) / (0.5*(rho(i)+rho(j)));
                PI_ij = PI_ij * 0.5 * (balsara(i) + balsara(j));
            end
            
            d_vel(i,:) = d_vel(i,:) - m0 * (P(i)/rho(i)^2 + P(j)/rho(j)^2 + PI_ij) * gradW_orig';
            
            Sj = reshape(S(j,:), 2, 2);
            acc_stress = (m0 * (Si/rho(i)^2 + Sj/rho(j)^2) * gradW_orig)'; 
            d_vel(i,:) = d_vel(i,:) + acc_stress;
            
            gradV = gradV - (m0/rho(j)) * (v_ij' * gradW_corr');
            d_u(i) = d_u(i) + 0.5 * m0 * PI_ij * dot_rv * (gradW_val/dist);
        end
        
        eps_dot = 0.5 * (gradV + gradV');
        omega = 0.5 * (gradV - gradV');
        trace_eps = trace(eps_dot);
        
        dS_elastic = 2 * mu * (eps_dot - 0.5 * trace_eps * eye(2));
        dS_rot = Si * omega' + Si' * omega; 
        d_S(i,:) = reshape(dS_elastic + dS_rot, 1, 4);
        
        d_u(i) = d_u(i) + (P(i)/rho(i)^2) * d_rho(i) + (1/rho(i)) * sum(sum(Si .* eps_dot));
    end
    dY = [d_pos(:); d_vel(:); d_rho; d_u; d_S(:)];
end

% --- 网格填充生成函数 ---
function [pos, vel] = create_ring_grid(spacing, r_in, r_out, center, v0)
    pos = [];
    % 生成一个覆盖环形的矩形网格
    x_range = -r_out:spacing:r_out;
    y_range = -r_out:spacing:r_out;
    for x = x_range
        for y = y_range
            d = sqrt(x^2 + y^2);
            if d >= r_in && d <= r_out
                pos = [pos; [x, y] + center];
            end
        end
    end
    vel = repmat(v0, size(pos, 1), 1);
end

% --- Cubic Spline Kernel ---
function [W, gradW] = cubic_spline_kernel(r, h)
    q = r / h;
    sigma = 10 / (7 * pi * h^2);
    if q < 1
        W = sigma * (1 - 1.5*q^2 + 0.75*q^3);
        gradW = sigma * (1/h) * (-3*q + 2.25*q^2);
    elseif q < 2
        W = sigma * 0.25 * (2 - q)^3;
        gradW = -sigma * 0.75 * (1/h) * (2 - q)^2;
    else
        W = 0; gradW = 0;
    end
end
function status = sph_monitor(t, ~, flag)
    persistent last_t
    if isempty(last_t)
        last_t = 0;
    end
    
    status = 0;
    if isempty(flag) % flag 为空表示正在积分步骤中
        % 计算当前时间步长 dt
        dt = t - last_t;
        
        % 假设总时间在 tspan 中定义（这里硬编码为 1.2，或者你可以通过全局变量获取）
        t_final = 1.2; 
        percentage = (t / t_final) * 100;
        
        % 打印进度 (每步都打印可能太快，可以根据需要加计数器限制打印频率)
        fprintf('%6.4f    %6.2f%%      %6.2e\n', t, percentage, dt);
        
        last_t = t;
    elseif strcmp(flag, 'init')
        last_t = 0;
    end
end