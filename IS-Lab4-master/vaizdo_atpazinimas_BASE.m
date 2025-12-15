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

%% Create the RBF neural network
%  num_neurons = number of RBF centers (3–12 recommended)
num_neurons = 12;  
% Notes:
%   12 neurons → works perfectly on all training and test cases
%   11 neurons → error in training data (predicts "2" instead of "$")
%   ≤10 neurons → errors increase (especially with "$"), more mismatches appear
tinklas = newrb(P, T, 0, 1, num_neurons);

%% ================================================================
%  VALIDATION USING A PART OF TRAINING DATA 
%  Extract some training samples to validate the recognizer

% Select columns 14–26 (second row of symbols) as a mini test
P2 = P(:,14:26);

% Feed the selected symbols into the trained network
Y2 = sim(tinklas, P2);

% Get predicted class index for each symbol
[~, b2] = max(Y2);

% Convert class indices → actual characters
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

disp(atsakymas);
figure(7), text(0.1,0.5,atsakymas,'FontSize',38);


%% ================================================================
%  TESTING ON EXTERNAL IMAGES
%  Each test block loads a new image, extracts features, and predicts text

%%  TEST IMAGE 1

pavadinimas = 'test_data1.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = sim(tinklas, P2);
[~, b2] = max(Y2);

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

disp(atsakymas);
figure(8), text(0.1,0.5,atsakymas,'FontSize',38), axis off;


%%  TEST IMAGE 2

pavadinimas = 'test_data2.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = sim(tinklas, P2);
[~, b2] = max(Y2);

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

disp(atsakymas);
figure(9), text(0.1,0.5,atsakymas,'FontSize',38), axis off;



%%  TEST IMAGE 3

pavadinimas = 'test_data3.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

P2 = cell2mat(pozymiai_patikrai);
Y2 = sim(tinklas, P2);
[~, b2] = max(Y2);

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

disp(atsakymas);
figure(10), text(0.1,0.5,atsakymas,'FontSize',38), axis off;
