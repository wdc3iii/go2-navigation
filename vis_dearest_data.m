clear
load("nearest_data.mat")

fprintf("Distance: min %0.2f, max %0.2f\n", min(nearest_dists, [], 'all'), max(nearest_dists, [], 'all'))

disp(min(nearest_inds(1, :, :), [], 'all'))
disp(max(nearest_inds(1, :, :), [], 'all'))
disp(min(nearest_inds(2, :, :), [], 'all'))
disp(max(nearest_inds(2, :, :), [], 'all'))

fprintf("Sub Map Origin: %0.2f, %0.2f\nSub map Theta:  %0.2f\n", map_origin(1), map_origin(2), map_theta)

figure(1);
clf
imshow(nearest_dists' / max(nearest_dists, [], 'all'))

%% Constraints
clear; clc;
load("constraints.mat")

figure(2)
clf
plot(zwarm(:, 1), zwarm(:, 2), '-o')
hold on
c = 1;
for i = 1:size(A, 1)
    a0 = A(i, :);
    b0 = b(i);

    p = nearest_points(i, :);
    plot(p(1), p(2), 'ok')
    if abs(a0(1)) > 0.5
        y = linspace(p(2) - c, p(2) + c);
        x = -(a0(2) * y + b0) / a0(1);
    else
        x = linspace(p(1) - c, p(1) + c);
        y = -(a0(1) * x + b0) / a0(2);
        plot(x, y)
    end
    plot(x, y)
end
d = 3;
% xlim([zwarm(1, 1) - d, zwarm(1, 1) + d])
% ylim([zwarm(1, 2) - d, zwarm(1, 2) + d])