%% Display the solve data for the MPC algorithm
clear; clc;

sol_files = dir('dtmpc_solves/*.mat');
maj_min = zeros(length(sol_files), 2);
for i = 1:length(sol_files)
    % Extract numbers using regular expressions
    tokens = regexp(sol_files(i).name, 'sol_(\d+)_(\d+)\.mat', 'tokens');
    if ~isempty(tokens)
        maj_min(i, :) = str2double(tokens{1});
    end
end
[~, idx] = sortrows(maj_min, [1, 2]);
sorted_sol_files = sol_files(idx);

sym_ref = '.-';
c_ref = [0.6350 0.0780 0.1840 1];
c_ref_old = [0.6350 0.0780 0.1840 0.5];
sym_init = '.-';
c_init = [0 0.4470 0.7410 0.5];
sym_sol = '.-';
c_sol = [0 0 0 1];
c_sol_old = [0 0 0 0.5];
tube_color = [0 0 0];
tube_alpha = 0.2;
c_con = [0.3010 0.7450 0.9330 0.5];
c_len = 0.2;

dt = 0.1;

set(groot, 'DefaultAxesFontSize', 17); % Set default font size for axes labels and tick
set(groot, 'DefaultTextFontSize', 17); % Set default font size for text objects 
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex'); % Set interpreter for axis tick labels 
set(groot, 'DefaultTextInterpreter', 'latex'); % Set interpreter for text objects (e.g., titles, labels) 
set(groot, 'DefaultLegendInterpreter', 'latex'); 
set(groot, 'DefaultFigureRenderer', 'painters'); 
set(groot, 'DefaultLineLineWidth', 2);
set(groot, 'DefaultLineMarkerSize', 15);


%% Setup plots
load('dtmpc_solves/sol_0_0.mat')
n = size(z, 2);
m = size(v, 2);
N = size(v, 1);
H = (size(params, 1) - n - (N + 1) * n - N * m - N * 2 - N - m - m) / (1 + m);
[z0, z_ref, v_ref, A, b, v_min, v_max, e, v_prev] = parse_params(params, n, m, N, H);
[z_init, v_init, w_init] = parse_init(x_init, n, m, N);

figure(1);
clf;
% Top left: plot spacial
subplot(4,2,[1, 3, 5, 7])
hold on
old_ref_plt = plot(z_ref(1, 1), z_ref(1, 2), sym_ref, 'Color', c_ref_old);
old_sol_plt = plot(z(:, 1), z(:, 2), sym_sol, 'Color', c_sol_old);
ref_plt = plot(z_ref(:, 1), z_ref(:, 2), sym_ref, 'Color', c_ref);

tubes = cell(N);
constraints = cell(N);
for i = 1:N
    r = max(0, w(i));
    tubes{i} = rectangle( ...
        'Position', [z(i + 1, 1) - r, z(i + 1, 2) - r, 2 * r, 2 * r], ...
        'Curvature', [1 1], 'EdgeColor', tube_color, 'FaceColor', tube_color, 'FaceAlpha', tube_alpha ...
    );
    a_i = A(i, :);
    b_i = b(i);

    p = proj_to_line(z(i + 1, 1:2), a_i, b_i);
    if abs(a_i(1)) > 0.5
        y = linspace(p(2) - c_len, p(2) + c_len);
        x = -(a_i(2) * y + b_i) / a_i(1);
    else
        x = linspace(p(1) - c_len, p(1) + c_len);
        y = -(a_i(1) * x + b_i) / a_i(2);
    end
    constraints{i} = plot(x, y, '-', 'Color', c_con);

end
init_plt = plot(z_init(:, 1), z_init(:, 2), sym_init, 'Color', c_init);
sol_plt = plot(z(:, 1), z(:, 2), sym_sol, 'Color', c_sol);
spacial_xlim = xlim;
spacial_ylim = ylim;

axis equal
xlabel("X (m)")
ylabel('Y (m)')
% legend('Reference', 'Initialization', 'Solution')
% Top Right: Plot trajectories
subplot(4, 2, 2)
hold on;
t0 = [0:N] * dt;
old_traj_x = plot(t0(1), z_ref(1, 1), sym_ref, 'Color', c_ref_old);
pld_sol_traj_x = plot(t0(1), z(1, 1), sym_sol, 'Color', c_sol_old);
ref_traj_x = plot(t0, z_ref(:, 1), sym_ref, 'Color', c_ref);
init_traj_x = plot(t0, z_init(:, 1), sym_init, 'Color', c_init);
sol_traj_x = plot(t0, z(:, 1), sym_sol, 'Color', c_sol);
xlabel("Time (s)")
ylabel('X (m)')
title('X Position')
legend('Reference', 'Initialization', 'Solution')

subplot(4, 2, 4)
hold on;
old_ref_traj_y = plot(t0(1), z_ref(1, 2), sym_ref, 'Color', c_ref_old);
old_sol_traj_y = plot(t0(1), z(1, 2), sym_sol, 'Color', c_sol_old);
ref_traj_y = plot(t0, z_ref(:, 2), sym_ref, 'Color', c_ref);
init_traj_y = plot(t0, z_init(:, 2), sym_init, 'Color', c_init);
sol_traj_y = plot(t0, z(:, 2), sym_sol, 'Color', c_sol);
xlabel("Time (s)")
ylabel('Y (m)')
legend('Reference', 'Initialization', 'Solution')
title('Y Position')

subplot(4, 2, 6)
hold on
old_ref_traj_vx = plot(t0(1), v_ref(1, 1), sym_ref, 'Color', c_ref_old);
old_sol_traj_vx = plot(t0(1), v(1, 1), sym_sol, 'Color', c_sol_old);
ref_traj_vx = plot(t0(1:end-1), v_ref(:, 1), sym_ref, 'Color', c_ref);
init_traj_vx = plot(t0(1:end-1), v_init(:, 1), sym_init, 'Color', c_init);
sol_traj_vx = plot(t0(1:end-1), v(:, 1), sym_sol, 'Color', c_sol);
xlabel("Time (s)")
ylabel('v (m/s)')
legend('Reference', 'Initialization', 'Solution')
title('Velocities')

subplot(4, 2, 8)
hold on
old_ref_traj_vy = plot(t0(1), v_ref(1, 2), sym_ref, 'Color', c_ref_old);
old_sol_traj_vy = plot(t0(1), v(1, 2), sym_sol, 'Color', c_sol_old);
ref_traj_vy = plot(t0(1:end-1), v_ref(:, 2), sym_ref, 'Color', c_ref);
init_traj_vy = plot(t0(1:end-1), v_init(:, 2), sym_init, 'Color', c_init);
sol_traj_vy = plot(t0(1:end-1), v(:, 2), sym_sol, 'Color', c_sol);
xlabel("Time (s)")
ylabel('v (m/s)')
legend('Reference', 'Initialization', 'Solution')
title('Velocities')



%% Loop through time and iteration
k = 1;
while k <= length(sorted_sol_files)
    load(['dtmpc_solves/' sorted_sol_files(k).name])
    if k == 1
        z_ref_old = z_ref(1, :);
        v_ref_old = v_ref(1, :);
        z_old = z(1, :);
        v_old = v(1, :);
        t_old = 0;
    else
        if maj_min(idx(k), 1) ~= maj_min(idx(k - 1), 1)
            z_ref_old = [z_ref_old; z_ref(1, :)];
            z_old = [z_old; z(1, :)];
            v_ref_old = [v_ref_old; v_ref(1, :)];
            v_old = [v_old; v(1, :)];
            t_old = [t_old t_old(end) + dt];
        end
    end

    spacial_xlim = [
        min([spacial_xlim(1); z_ref(:, 1) - 0.2; z_init(:, 1) - 0.2; z(:, 1) - 0.2]), ...
        max([spacial_xlim(2); z_ref(:, 1) + 0.2; z_init(:, 1) + 0.2; z(:, 1) + 0.2])
    ];
    spacial_ylim = [
        min([spacial_ylim(1); z_ref(:, 2) - 0.2; z_init(:, 2) - 0.2; z(:, 2) - 0.2]), ...
        max([spacial_ylim(2); z_ref(:, 2) + 0.2; z_init(:, 2) + 0.2; z(:, 2) + 0.2])
    ];
    
    t = maj_min(idx(k), 1);
    [z0, z_ref, v_ref, A, b, v_min, v_max, e, v_prev] = parse_params(params, n, m, N, H);
    [z_init, v_init, w_init] = parse_init(x_init, n, m, N);

    ref_plt.XData = z_ref(:, 1);
    ref_plt.YData = z_ref(:, 2);
    init_plt.XData = z_init(:, 1);
    init_plt.YData = z_init(:, 2);
    sol_plt.XData = z(:, 1);
    sol_plt.YData = z(:, 2);

    ref_traj_x.YData = z_ref(:, 1);
    ref_traj_y.YData = z_ref(:, 2);
    init_traj_x.YData = z_init(:, 1);
    init_traj_y.YData = z_init(:, 2);
    sol_traj_x.YData = z(:, 1);
    sol_traj_y.YData = z(:, 2);

    ref_traj_vx.YData = v_ref(:, 1);
    ref_traj_vy.YData = v_ref(:, 2);
    init_traj_vx.YData = v_init(:, 1);
    init_traj_vy.YData = v_init(:, 2);
    sol_traj_vx.YData = v(:, 1);
    sol_traj_vy.YData = v(:, 2);

    ref_traj_x.XData = t * dt + t0;
    ref_traj_y.XData = t * dt + t0;
    init_traj_x.XData = t * dt + t0;
    init_traj_y.XData = t * dt + t0;
    sol_traj_x.XData = t * dt + t0;
    sol_traj_y.XData = t * dt + t0;

    ref_traj_vx.XData = t * dt + t0(1:end-1);
    ref_traj_vy.XData = t * dt + t0(1:end-1);
    init_traj_vx.XData = t * dt + t0(1:end-1);
    init_traj_vy.XData = t * dt + t0(1:end-1);
    sol_traj_vx.XData = t * dt + t0(1:end-1);
    sol_traj_vy.XData = t * dt + t0(1:end-1);

    old_ref_plt.XData = z_ref_old(:, 1);
    old_ref_plt.YData = z_ref_old(:, 2);
    old_sol_plt.XData = z_old(:, 1);
    old_sol_plt.YData = z_old(:, 2);

    old_ref_traj_x.YData = z_ref_old(:, 1);
    old_ref_traj_y.YData = z_ref_old(:, 2);
    old_sol_traj_x.YData = z_old(:, 1);
    old_sol_traj_y.YData = z_old(:, 2);

    old_ref_traj_vx.YData = v_ref_old(:, 1);
    old_ref_traj_vy.YData = v_ref_old(:, 2);
    old_sol_traj_vx.YData = v_old(:, 1);
    old_sol_traj_vy.YData = v_old(:, 2);

    old_ref_traj_x.XData = t_old;
    old_ref_traj_y.XData = t_old;
    old_sol_traj_x.XData = t_old;
    old_sol_traj_y.XData = t_old;

    old_ref_traj_vx.XData = t_old;
    old_ref_traj_vy.XData = t_old;
    old_sol_traj_vx.XData = t_old;
    old_sol_traj_vy.XData = t_old;

    subplot(4, 2, [1, 3, 5, 7])
    xlim(spacial_xlim)
    ylim(spacial_ylim)
    
    for i = 1:N
        r = max(0, w(i));
        tubes{i}.Position = [z(i + 1, 1) - r, z(i + 1, 2) - r, 2 * r, 2 * r];

        a_i = A(i, :);
        b_i = b(i);
    
        p = proj_to_line(z(i + 1, 1:2), a_i, b_i);
        if abs(a_i(1)) > 0.5
            y = linspace(p(2) - c_len, p(2) + c_len);
            x = -(a_i(2) * y + b_i) / a_i(1);
        else
            x = linspace(p(1) - c_len, p(1) + c_len);
            y = -(a_i(1) * x + b_i) / a_i(2);
        end
        constraints{i}.XData = x;
        constraints{i}.YData = y;
    end
    k = k + 1;
    pause(0.01)
end


function [z0, z_ref, v_ref, A, b, v_min, v_max, e, v_prev] = parse_params(params, n, m, N, H)
i = 1;
z0 = params(i:i + n - 1);
i = i + n;
z_ref = reshape(params(i:i + (N + 1) * n - 1), N + 1, n);
i = i + (N + 1) * n;
v_ref = reshape(params(i:i + N * m - 1), N, m);
i = i + N * m;
A = reshape(params(i:i + N * 2 - 1), N, 2);
i = i + N * 2;
b = params(i:i + N - 1);
i = i + N;
v_min = params(i:i + m - 1);
i = i + m;
v_max = params(i:i + m - 1);
i = i + m;
e = params(i:i + H - 1);
i = i + H;
v_prev = reshape(params(i:i + H * m - 1), H, m);
assert(i + H * m - 1 == size(params, 1))
end

function [z_init, v_init, w_init] = parse_init(x_init, n, m, N)
i = 1;
z_init = reshape(x_init(i:i + (N + 1) * n - 1), N + 1, n);
i = i + (N + 1) * n;
v_init = reshape(x_init(i:i + N * m - 1), N, m);
i = i + N * m;
w_init = x_init(i:i + N - 1);
assert(i + N - 1 == size(x_init, 1))
end

function p_proj = proj_to_line(p, a, b)
if abs(a(2)) > 0.5
    p1 = [0, -b / a(2)];
    p2 = [1, (-b - a(1)) / a(2)];
else
    p1 = [-b / a(1), 0];
    p2 = [(-b - a(2)) / a(1), 1];
end
p12 = p2 - p1;
t = dot(p - p1, p12) / dot(p12, p12);
p_proj = p1 + t * p12;
end