% Naive Bayes Classifier for Apple and Pear Classification

% Clear workspace
clear; clc; close all;

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

% Calculate features for each image
% For Apples
features_apple = [];
features_apple(1,:) = [spalva_color(A1), apvalumas_roundness(A1)];
features_apple(2,:) = [spalva_color(A2), apvalumas_roundness(A2)];
features_apple(3,:) = [spalva_color(A3), apvalumas_roundness(A3)];
features_apple(4,:) = [spalva_color(A4), apvalumas_roundness(A4)];
features_apple(5,:) = [spalva_color(A5), apvalumas_roundness(A5)];
features_apple(6,:) = [spalva_color(A6), apvalumas_roundness(A6)];
features_apple(7,:) = [spalva_color(A7), apvalumas_roundness(A7)];
features_apple(8,:) = [spalva_color(A8), apvalumas_roundness(A8)];
features_apple(9,:) = [spalva_color(A9), apvalumas_roundness(A9)];

% For Pears
features_pear = [];
features_pear(1,:) = [spalva_color(P1), apvalumas_roundness(P1)];
features_pear(2,:) = [spalva_color(P2), apvalumas_roundness(P2)];
features_pear(3,:) = [spalva_color(P3), apvalumas_roundness(P3)];
features_pear(4,:) = [spalva_color(P4), apvalumas_roundness(P4)];

% Create training data (using first 3 apples and first 2 pears as in perceptron example)
train_features = [features_apple(1:3,:); features_pear(1:2,:)];
train_labels = [1; 1; 1; -1; -1]; % 1 for apple, -1 for pear
train_images = {'A1', 'A2', 'A3', 'P1', 'P2'};

% Create test data (remaining images)
test_features = [features_apple(4:9,:); features_pear(3:4,:)];
test_labels = [1; 1; 1; 1; 1; 1; -1; -1];
test_images = {'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'P3', 'P4'};

fprintf('=== Naive Bayes Classifier Implementation ===\n\n');

% Calculate prior probabilities
n_apple = sum(train_labels == 1);
n_pear = sum(train_labels == -1);
n_total = length(train_labels);

P_apple = n_apple / n_total;
P_pear = n_pear / n_total;

fprintf('Prior probabilities:\n');
fprintf('P(Apple) = %f\n', P_apple);
fprintf('P(Pear) = %f\n\n', P_pear);

% Calculate mean and standard deviation for each feature and class
% For apples (class 1)
apple_features = train_features(train_labels == 1, :);
mu_apple_color = mean(apple_features(:, 1));
mu_apple_roundness = mean(apple_features(:, 2));
sigma_apple_color = std(apple_features(:, 1));
sigma_apple_roundness = std(apple_features(:, 2));

% For pears (class -1)
pear_features = train_features(train_labels == -1, :);
mu_pear_color = mean(pear_features(:, 1));
mu_pear_roundness = mean(pear_features(:, 2));
sigma_pear_color = std(pear_features(:, 1));
sigma_pear_roundness = std(pear_features(:, 2));

fprintf('Class statistics:\n');
fprintf('Apples - Color: μ=%.4f, σ=%.4f\n', mu_apple_color, sigma_apple_color);
fprintf('Apples - Roundness: μ=%.4f, σ=%.4f\n', mu_apple_roundness, sigma_apple_roundness);
fprintf('Pears - Color: μ=%.4f, σ=%.4f\n', mu_pear_color, sigma_pear_color);
fprintf('Pears - Roundness: μ=%.4f, σ=%.4f\n\n', mu_pear_roundness, sigma_pear_roundness);

% Gaussian probability density function
gaussian_pdf = @(x, mu, sigma) (1/(sqrt(2*pi)*sigma)) * exp(-(x-mu)^2/(2*sigma^2));

% Test on training data
fprintf('=== Training Data Results ===\n');
correct_train = 0;
for i = 1:length(train_labels)
    color_val = train_features(i, 1);
    roundness_val = train_features(i, 2);
    
    % Calculate likelihood for apple class
    like_apple_color = gaussian_pdf(color_val, mu_apple_color, sigma_apple_color);
    like_apple_roundness = gaussian_pdf(roundness_val, mu_apple_roundness, sigma_apple_roundness);
    posterior_apple = P_apple * like_apple_color * like_apple_roundness;
    
    % Calculate likelihood for pear class
    like_pear_color = gaussian_pdf(color_val, mu_pear_color, sigma_pear_color);
    like_pear_roundness = gaussian_pdf(roundness_val, mu_pear_roundness, sigma_pear_roundness);
    posterior_pear = P_pear * like_pear_color * like_pear_roundness;
    
    % Make prediction
    if posterior_apple > posterior_pear
        prediction = 1;
    else
        prediction = -1;
    end
    
    correct = prediction == train_labels(i);
    if correct
        correct_train = correct_train + 1;
    end
    
    fprintf('Sample %s: True=%d, Pred=%d, Correct=%d\n', ...
        train_images{i}, train_labels(i), prediction, correct);
end
fprintf('Training Accuracy: %.2f%% (%d/%d)\n\n', 100*correct_train/length(train_labels), correct_train, length(train_labels));

% Test on test data
fprintf('=== Test Data Results ===\n');
correct_test = 0;
predictions = zeros(length(test_labels), 1);

for i = 1:length(test_labels)
    color_val = test_features(i, 1);
    roundness_val = test_features(i, 2);
    
    % Calculate likelihood for apple class
    like_apple_color = gaussian_pdf(color_val, mu_apple_color, sigma_apple_color);
    like_apple_roundness = gaussian_pdf(roundness_val, mu_apple_roundness, sigma_apple_roundness);
    posterior_apple = P_apple * like_apple_color * like_apple_roundness;
    
    % Calculate likelihood for pear class
    like_pear_color = gaussian_pdf(color_val, mu_pear_color, sigma_pear_color);
    like_pear_roundness = gaussian_pdf(roundness_val, mu_pear_roundness, sigma_pear_roundness);
    posterior_pear = P_pear * like_pear_color * like_pear_roundness;
    
    % Make prediction
    if posterior_apple > posterior_pear
        prediction = 1;
        class_name = 'Apple';
    else
        prediction = -1;
        class_name = 'Pear';
    end
    
    predictions(i) = prediction;
    correct = prediction == test_labels(i);
    if correct
        correct_test = correct_test + 1;
    end
    
    true_class = test_labels(i);
    true_class_name = 'Apple';
    if true_class == -1
        true_class_name = 'Pear';
    end
    
    fprintf('Sample %s: True=%s, Pred=%s, Color=%.4f, Roundness=%.4f, Correct=%d\n', ...
            test_images{i}, true_class_name, class_name, color_val, roundness_val, correct);
end
fprintf('Test Accuracy: %.2f%% (%d/%d)\n\n', 100*correct_test/length(test_labels), correct_test, length(test_labels));

% Calculate and display confusion matrix
fprintf('=== Confusion Matrix ===\n');
TP = sum((predictions == 1) & (test_labels == 1));
FP = sum((predictions == 1) & (test_labels == -1));
TN = sum((predictions == -1) & (test_labels == -1));
FN = sum((predictions == -1) & (test_labels == 1));

fprintf('            Predicted Apple  Predicted Pear\n');
fprintf('True Apple      %d               %d\n', TP, FN);
fprintf('True Pear       %d               %d\n\n', FP, TN);

% % Calculate additional metrics
% accuracy = (TP + TN) / length(test_labels);
% precision = TP / (TP + FP);
% recall = TP / (TP + FN);
% f1_score = 2 * (precision * recall) / (precision + recall);
% 
% fprintf('Performance Metrics:\n');
% fprintf('Accuracy:  %.3f\n', accuracy);
% fprintf('Precision: %.3f\n', precision);
% fprintf('Recall:    %.3f\n', recall);
% fprintf('F1-Score:  %.3f\n', f1_score);

% Visualization of the decision boundary (made by AI just for studying purposes)
fprintf('\n=== Generating Decision Boundary Visualization ===\n');

% Create a grid for visualization
color_range = linspace(min([features_apple(:,1); features_pear(:,1)])-0.1, ...
                      max([features_apple(:,1); features_pear(:,1)])+0.1, 50);
roundness_range = linspace(min([features_apple(:,2); features_pear(:,2)])-0.1, ...
                          max([features_apple(:,2); features_pear(:,2)])+0.1, 50);

[Color, Roundness] = meshgrid(color_range, roundness_range);
grid_points = [Color(:), Roundness(:)];

% Classify each point in the grid
grid_predictions = zeros(size(grid_points, 1), 1);
for i = 1:size(grid_points, 1)
    color_val = grid_points(i, 1);
    roundness_val = grid_points(i, 2);
    
    like_apple_color = gaussian_pdf(color_val, mu_apple_color, sigma_apple_color);
    like_apple_roundness = gaussian_pdf(roundness_val, mu_apple_roundness, sigma_apple_roundness);
    posterior_apple = P_apple * like_apple_color * like_apple_roundness;
    
    like_pear_color = gaussian_pdf(color_val, mu_pear_color, sigma_pear_color);
    like_pear_roundness = gaussian_pdf(roundness_val, mu_pear_roundness, sigma_pear_roundness);
    posterior_pear = P_pear * like_pear_color * like_pear_roundness;
    
    if posterior_apple > posterior_pear
        grid_predictions(i) = 1;
    else
        grid_predictions(i) = -1;
    end
end

% Plot the results - SIMPLE AND ROBUST
figure;
decision_map = reshape(grid_predictions, size(Color));

% Plot decision regions using imagesc (most compatible)
imagesc(color_range, roundness_range, decision_map);
colormap([1 0.8 0.8; 0.8 1 0.8]); % Light red for apple, light green for pear
set(gca, 'YDir', 'normal'); % Important for correct axis orientation
hold on;

% Plot data points
plot(features_apple(1:3,1), features_apple(1:3,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Training Apples');
plot(features_pear(1:2,1), features_pear(1:2,2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Training Pears');
plot(features_apple(4:9,1), features_apple(4:9,2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Test Apples');
plot(features_pear(3:4,1), features_pear(3:4,2), 'g^', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Test Pears');

xlabel('Color Feature (HSV)');
ylabel('Roundness Feature');
title('Naive Bayes Classifier - Decision Boundary');
legend('Location', 'best');
grid on;

fprintf('Visualization complete. Check the generated plot.\n');
