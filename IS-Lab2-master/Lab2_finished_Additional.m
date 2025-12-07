clear all; close all; clc;

%% DATA GENERATION
% 400 samples, x from 0 to 1
[x1, x2] = meshgrid(linspace(0.1,1,20), linspace(0.1,1,20));
x1 = x1(:);
x2 = x2(:);

% Desired function (target)
y_target = (1 + 0.6 * sin(2*pi*x1/0.7) + 0.3*sin(2*pi*x2)) / 2;
%y_target = y_target / 2; % For amplitude minimization

%% Activation help functions
tanh_der = @(v) 1 - tanh(v).^2;

%% WEIGHT INITIALIZATION
% Example of indexing: W21 it goes TO 2nd output FROM 1st input

% Hidden layer

% Hidden weights
w11_1 = randn(1);
w21_1 = randn(1);
w31_1 = randn(1);
w41_1 = randn(1);
w51_1 = randn(1);
w61_1 = randn(1);
w71_1 = randn(1);
w81_1 = randn(1);

w12_1 = randn(1);
w22_1 = randn(1);
w32_1 = randn(1);
w42_1 = randn(1);
w52_1 = randn(1);
w62_1 = randn(1);
w72_1 = randn(1);
w82_1 = randn(1);

% Hidden bais
b1_1 = randn(1);
b2_1 = randn(1);
b3_1 = randn(1);
b4_1 = randn(1);
b5_1 = randn(1);
b6_1 = randn(1);
b7_1 = randn(1);
b8_1 = randn(1);

% Output layer

% Weights
w11_2 = randn(1);
w12_2 = randn(1);
w13_2 = randn(1);
w14_2 = randn(1);
w15_2 = randn(1);
w16_2 = randn(1);
w17_2 = randn(1);
w18_2 = randn(1);

% Bias
b1_2 = randn(1);

%% Some Parameters
eta = 0.01;
epochs = 100000;
errors = zeros(epochs,1);

%% Training Loop

for epoch = 1 : epochs

    total_err = 0;

    for i = 1:length(x1)
        % Hidden layer
        % Weighted sums
        v1_1 = x1(i)*w11_1 + x2(i)*w12_1 + b1_1;
        v2_1 = x1(i)*w21_1 + x2(i)*w22_1 + b2_1;
        v3_1 = x1(i)*w31_1 + x2(i)*w32_1 + b3_1;
        v4_1 = x1(i)*w41_1 + x2(i)*w42_1 + b4_1;
        v5_1 = x1(i)*w51_1 + x2(i)*w52_1 + b5_1;
        v6_1 = x1(i)*w61_1 + x2(i)*w62_1 + b6_1;
        v7_1 = x1(i)*w71_1 + x2(i)*w72_1 + b7_1;
        v8_1 = x1(i)*w81_1 + x2(i)*w82_1 + b8_1;
    
        % Activation
        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);
        y5_1 = tanh(v5_1);
        y6_1 = tanh(v6_1);
        y7_1 = tanh(v7_1);
        y8_1 = tanh(v8_1);
        
        % Output layer
        % Weighted sum
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + ...
            y5_1*w15_2 + y6_1*w16_2 + y7_1*w17_2 + y8_1*w18_2 + b1_2;
        % Activation
        y1_2 = v1_2;
        % Final output
        y = y1_2;
    
        % Backpropogation
        % Error
        e = y_target(i) - y ;
        
        total_err = total_err + e^2;

        % Error gradient for output layer
        delta1_2 = e;
        % Error gradient for hidden layer
        delta1_1 = tanh_der(v1_1) * delta1_2 * w11_2;
        delta2_1 = tanh_der(v2_1) * delta1_2 * w12_2;
        delta3_1 = tanh_der(v3_1) * delta1_2 * w13_2;
        delta4_1 = tanh_der(v4_1) * delta1_2 * w14_2;
        delta5_1 = tanh_der(v5_1) * delta1_2 * w15_2;
        delta6_1 = tanh_der(v6_1) * delta1_2 * w16_2;
        delta7_1 = tanh_der(v7_1) * delta1_2 * w17_2;
        delta8_1 = tanh_der(v8_1) * delta1_2 * w18_2;
        
        % Update weights and bias | Output layer
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;
        w15_2 = w15_2 + eta*delta1_2*y5_1;
        w16_2 = w16_2 + eta*delta1_2*y6_1;
        w17_2 = w17_2 + eta*delta1_2*y7_1;
        w18_2 = w18_2 + eta*delta1_2*y8_1;
    
        b1_2 = b1_2 + eta*delta1_2;
    
        % Update weights and bias | Hidden layer
        w11_1 = w11_1 + eta*delta1_1*x1(i);
        w21_1 = w21_1 + eta*delta2_1*x1(i);
        w31_1 = w31_1 + eta*delta3_1*x1(i);
        w41_1 = w41_1 + eta*delta4_1*x1(i);
        w51_1 = w51_1 + eta*delta5_1*x1(i);
        w61_1 = w61_1 + eta*delta6_1*x1(i);
        w71_1 = w71_1 + eta*delta7_1*x1(i);
        w81_1 = w81_1 + eta*delta8_1*x1(i);

        w12_1 = w12_1 + eta*delta1_1*x2(i);
        w22_1 = w22_1 + eta*delta2_1*x2(i);
        w32_1 = w32_1 + eta*delta3_1*x2(i);
        w42_1 = w42_1 + eta*delta4_1*x2(i);
        w52_1 = w52_1 + eta*delta5_1*x2(i);
        w62_1 = w62_1 + eta*delta6_1*x2(i);
        w72_1 = w72_1 + eta*delta7_1*x2(i);
        w82_1 = w82_1 + eta*delta8_1*x2(i);
    
        b1_1 = b1_1 + eta*delta1_1;
        b2_1 = b2_1 + eta*delta2_1;
        b3_1 = b3_1 + eta*delta3_1;
        b4_1 = b4_1 + eta*delta4_1;
        b5_1 = b5_1 + eta*delta5_1;
        b6_1 = b6_1 + eta*delta6_1;
        b7_1 = b7_1 + eta*delta7_1;
        b8_1 = b8_1 + eta*delta8_1;
    
    end

    errors(epoch) = total_err / length(x1);
end

%% Final loop with updated parameters for test samples
[x1_test, x2_test] = meshgrid(linspace(0.1,1,65), linspace(0.1,1,65));
x1_test = x1_test(:);
x2_test = x2_test(:);
y_pred = zeros(size(x1_test));

for i = 1:length(x1_test)
    % Hidden layer
    % Weighted sums
    v1_1 = x1_test(i)*w11_1 + x2_test(i)*w12_1 + b1_1;
    v2_1 = x1_test(i)*w21_1 + x2_test(i)*w22_1 + b2_1;
    v3_1 = x1_test(i)*w31_1 + x2_test(i)*w32_1 + b3_1;
    v4_1 = x1_test(i)*w41_1 + x2_test(i)*w42_1 + b4_1;
    v5_1 = x1_test(i)*w51_1 + x2_test(i)*w52_1 + b5_1;
    v6_1 = x1_test(i)*w61_1 + x2_test(i)*w62_1 + b6_1;
    v7_1 = x1_test(i)*w71_1 + x2_test(i)*w72_1 + b7_1;
    v8_1 = x1_test(i)*w81_1 + x2_test(i)*w82_1 + b8_1;

    % Activation
    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);
    y5_1 = tanh(v5_1);
    y6_1 = tanh(v6_1);
    y7_1 = tanh(v7_1);
    y8_1 = tanh(v8_1);
    
    % Output layer
    % Weighted sum
    v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + ...
        y5_1*w15_2 + y6_1*w16_2 + y7_1*w17_2 + y8_1*w18_2 + b1_2;
    % Activation
    y1_2 = v1_2;
    % Final output
    y_pred(i) = y1_2;

end


%% Plots
figure;
surf(reshape(x1,20,20), reshape(x2,20,20), reshape(y_target,20,20));
title('Target Surface');
xlabel('x1'); ylabel('x2'); zlabel('y');

figure;
surf(reshape(x1_test,65,65), reshape(x2_test,65,65), reshape(y_pred,65,65));
title('MLP Output Surface');
xlabel('x1'); ylabel('x2'); zlabel('y');

figure; 
plot(errors, 'LineWidth', 1.5); 
title('Training Error');
xlabel('Epoch'); 
ylabel('MSE');
grid on;