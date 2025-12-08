% Clear everything
clear; clc; close all;

%% Generate training data
x = linspace(0.1, 1, 20)';
x_test = linspace(0.1, 1, 65)';

y = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;

% Length of the vector x and x_test
N = length(x);
N_test = length(x_test);

% Manually chosen RBF centers/radii
c1 = 0.2; r1 = 0.15;
c2 = 0.9; r2 = 0.15;

% Initialize weights
w1 = rand(1); 
w2 = rand(1); 
w0 = rand(1);

eta = 0.05;    % learning rate
eta_c = 0.01;      % center learning rate
eta_r = 0.01;      % radius learning rate
epochs = 5000;  % number of training passes

%% Training loop (perceptron)
for epoch = 1:epochs
    for n = 1:N
        
        % RBF activations
        F1 = exp(-(x(n)-c1)^2/(2*r1^2));
        F2 = exp(-(x(n)-c2)^2/(2*r2^2));
        
        % Network output
        v = w1*F1 + w2*F2 + w0;
        
        % Error
        e = y(n) - v;
        
        % Weight updates (delta rule)
        w1 = w1 + eta * e * F1;
        w2 = w2 + eta * e * F2;
        w0 = w0 + eta * e;

        % Update centers
        dc1 = -e * w1 * F1 * (x(n)-c1) / (r1^2);
        dc2 = -e * w2 * F2 * (x(n)-c2) / (r2^2);
        
        c1 = c1 - eta_c * dc1;
        c2 = c2 - eta_c * dc2;
        
        %  Update radii
        dr1 = -e * w1 * F1 * ((x(n)-c1)^2) / (r1^3);
        dr2 = -e * w2 * F2 * ((x(n)-c2)^2) / (r2^3);
        
        r1 = r1 - eta_r * dr1;
        r2 = r2 - eta_r * dr2;
    end
end

%% Evaluate network

for n = 1:N_test
    F1 = exp(-(x_test(n)-c1)^2/(2*r1^2));
    F2 = exp(-(x_test(n)-c2)^2/(2*r2^2));
    y_est(n) = w1*F1 + w2*F2 + w0;
end

%% Plot results
plot(x, y, 'b.-', 'LineWidth', 3, 'MarkerSize', 25); hold on;
plot(x_test, y_est, 'r.--', 'LineWidth', 3, 'MarkerSize', 25);
legend("Target y", "RBF approximation");
title("RBF Network with Perceptron Training");
grid on;
