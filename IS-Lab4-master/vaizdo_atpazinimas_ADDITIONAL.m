close all; clear all; clc;

%% ================================================================
%  TRAINING STAGE
%  Load training image, extract features, create targets, train RBF network


%% Load the image containing handwritten training symbols
%  'pavadinimas' = filename of your handwritten training data image
%  The second argument (10) = number of text lines in the training image
pavadinimas = 'train_data.png';
pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, 10);

%% Convert extracted features from cell array → numeric matrix
%  Each column = feature vector of one character
P = cell2mat(pozymiai_tinklo_mokymui);

%% Construct correct target classes for training
%  There are 13 different symbols in total (1–9, 0, $, €, £)
%  For each row in the training image, we repeat identity matrix eye(13)
%  Because each of the 10 rows contains 13 characters
T = [eye(13), eye(13), eye(13), eye(13), eye(13), ...
     eye(13), eye(13), eye(13), eye(13), eye(13)];

%% ================================================================
%  ADDITIONAL TASK: MULTILAYER PERCEPTRON (MLP)

% Define the architecture:
% Example: two hidden layers with 30 and 20 neurons
% It can be configured (20,10), (40,20), or (50,30) for testing
hidden_layer_sizes = [30 20];

% Create MLP network
mlp_net = feedforwardnet(hidden_layer_sizes, 'trainlm');  % Levenberg–Marquardt training

% Train the network
mlp_net = train(mlp_net, P, T);

%% ================================================================
%  VALIDATION USING PART OF TRAINING DATA (same as with RBF)


P2 = P(:,14:26);          % second row
Y2 = mlp_net(P2);         % evaluate MLP
[~, b2] = max(Y2);        % predicted classes

raidziu_sk = size(P2,2);
atsakymas = [];

for k = 1:raidziu_sk
    switch b2(k)
        case 1,  atsakymas = [atsakymas, '1'];
        case 2,  atsakymas = [atsakymas, '2'];
        case 3,  atsakymas = [atsakymas, '3'];
        case 4,  atsakymas = [atsakymas, '4'];
        case 5,  atsakymas = [atsakymas, '5'];
        case 6,  atsakymas = [atsakymas, '6'];
        case 7,  atsakymas = [atsakymas, '7'];
        case 8,  atsakymas = [atsakymas, '8'];
        case 9,  atsakymas = [atsakymas, '9'];
        case 10, atsakymas = [atsakymas, '0'];
        case 11, atsakymas = [atsakymas, '$'];
        case 12, atsakymas = [atsakymas, '€'];
        case 13, atsakymas = [atsakymas, '£'];
    end
end

disp('MLP validation result:');
disp(atsakymas);
figure, text(0.1,0.5,atsakymas,'FontSize',38);


%% ================================================================
%  TESTING ON EXTERNAL IMAGES (MLP)


%% TEST IMAGE 1
pavadinimas = 'test_data1.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = mlp_net(P2);
[~, b2] = max(Y2);

atsakymas = [];
for k = 1:length(b2)
    switch b2(k)
        case 1,  atsakymas = [atsakymas, '1'];
        case 2,  atsakymas = [atsakymas, '2'];
        case 3,  atsakymas = [atsakymas, '3'];
        case 4,  atsakymas = [atsakymas, '4'];
        case 5,  atsakymas = [atsakymas, '5'];
        case 6,  atsakymas = [atsakymas, '6'];
        case 7,  atsakymas = [atsakymas, '7'];
        case 8,  atsakymas = [atsakymas, '8'];
        case 9,  atsakymas = [atsakymas, '9'];
        case 10, atsakymas = [atsakymas, '0'];
        case 11, atsakymas = [atsakymas, '$'];
        case 12, atsakymas = [atsakymas, '€'];
        case 13, atsakymas = [atsakymas, '£'];
    end
end
disp('MLP Test 1:');
disp(atsakymas);
figure, text(0.1,0.5,atsakymas,'FontSize',38), axis off;


%% TEST IMAGE 2
pavadinimas = 'test_data2.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = mlp_net(P2);
[~, b2] = max(Y2);

atsakymas = [];
for k = 1:length(b2)
    switch b2(k)
        case 1,  atsakymas = [atsakymas, '1'];
        case 2,  atsakymas = [atsakymas, '2'];
        case 3,  atsakymas = [atsakymas, '3'];
        case 4,  atsakymas = [atsakymas, '4'];
        case 5,  atsakymas = [atsakymas, '5'];
        case 6,  atsakymas = [atsakymas, '6'];
        case 7,  atsakymas = [atsakymas, '7'];
        case 8,  atsakymas = [atsakymas, '8'];
        case 9,  atsakymas = [atsakymas, '9'];
        case 10, atsakymas = [atsakymas, '0'];
        case 11, atsakymas = [atsakymas, '$'];
        case 12, atsakymas = [atsakymas, '€'];
        case 13, atsakymas = [atsakymas, '£'];
    end
end
disp('MLP Test 2:');
disp(atsakymas);
figure, text(0.1,0.5,atsakymas,'FontSize',38), axis off;


%% TEST IMAGE 3
pavadinimas = 'test_data3.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = mlp_net(P2);
[~, b2] = max(Y2);

atsakymas = [];
for k = 1:length(b2)
    switch b2(k)
        case 1,  atsakymas = [atsakymas, '1'];
        case 2,  atsakymas = [atsakymas, '2'];
        case 3,  atsakymas = [atsakymas, '3'];
        case 4,  atsakymas = [atsakymas, '4'];
        case 5,  atsakymas = [atsakymas, '5'];
        case 6,  atsakymas = [atsakymas, '6'];
        case 7,  atsakymas = [atsakymas, '7'];
        case 8,  atsakymas = [atsakymas, '8'];
        case 9,  atsakymas = [atsakymas, '9'];
        case 10, atsakymas = [atsakymas, '0'];
        case 11, atsakymas = [atsakymas, '$'];
        case 12, atsakymas = [atsakymas, '€'];
        case 13, atsakymas = [atsakymas, '£'];
    end
end
disp('MLP Test 3:');
disp(atsakymas);
figure, text(0.1,0.5,atsakymas,'FontSize',38), axis off;
