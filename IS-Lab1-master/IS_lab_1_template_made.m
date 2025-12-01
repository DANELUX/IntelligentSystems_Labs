clc; close all; clear all;

% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

% Set learning rate
eta = 0.1;

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% Initialize errors
e1 = 1; e2 = 1; e3 = 1; e4 = 1; e5 = 1;

% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);

% Training algorithm
iteration = 0;
max_iterations = 1000; % Prevent infinite loop

while e ~= 0 && iteration < max_iterations % executes while the total error is not 0
    iteration = iteration + 1;
    
    % Process each training example
    for i = 1:5
        % Get current input
        current_x1 = P(1, i);
        current_x2 = P(2, i);
        
        % Calculate weighted sum
        v = w1 * current_x1 + w2 * current_x2 + b;
        
        % Calculate output
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % Calculate error
        e_current = T(i) - y;
        
        % Update parameters
        w1 = w1 + eta * e_current * current_x1;
        w2 = w2 + eta * e_current * current_x2;
        b = b + eta * e_current;
    end
    
    % Test updated parameters on all training examples
    e1 = 0; e2 = 0; e3 = 0; e4 = 0; e5 = 0;
    
    for i = 1:5
        % Get current input
        current_x1 = P(1, i);
        current_x2 = P(2, i);
        
        % Calculate weighted sum
        v = w1 * current_x1 + w2 * current_x2 + b;
        
        % Calculate output
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % Calculate error
        error_current = T(i) - y;
        
        % Store error for each example
        if i == 1
            e1 = error_current;
        elseif i == 2
            e2 = error_current;
        elseif i == 3
            e3 = error_current;
        elseif i == 4
            e4 = error_current;
        elseif i == 5
            e5 = error_current;
        end
    end
    
    % Calculate total error
    e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);
    
    % Display progress every 100 iterations
    if mod(iteration, 100) == 0
        fprintf('Iteration %d, Total Error: %f\n', iteration, e);
    end
end

% Display final results
fprintf('Training completed after %d iterations\n', iteration);
fprintf('Final parameters: w1 = %f, w2 = %f, b = %f\n', w1, w2, b);
fprintf('Final total error: %f\n', e);

% Test the trained perceptron on all examples
fprintf('\nTesting the trained perceptron on trained examples:\n');
for i = 1:5
    current_x1 = P(1, i);
    current_x2 = P(2, i);
    
    v = w1 * current_x1 + w2 * current_x2 + b;
    
    if v > 0
        y = 1;
    else
        y = -1;
    end
    
    fprintf('Example %d: Desired = %d, Actual = %d, Correct = %d\n', ...
            i, T(i), y, T(i) == y);
end

x3 = [hsv_value_A4, hsv_value_A5, hsv_value_A6, hsv_value_P3, hsv_value_P4];
x4 = [metric_A4 metric_A5 metric_A6 metric_P3 metric_P4];
P2=[x3;x4];

% Test the trained perceptron on other examples
fprintf('\nTesting the trained perceptron on other examples:\n');
for i = 1:5
    current_x1 = P2(1, i);
    current_x2 = P2(2, i);
    
    v = w1 * current_x1 + w2 * current_x2 + b;
    
    if v > 0
        y = 1;
    else
        y = -1;
    end
    
    fprintf('Example %d: Desired = %d, Actual = %d, Correct = %d\n', ...
            i, T(i), y, T(i) == y);
end

