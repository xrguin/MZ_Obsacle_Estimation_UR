%% Test APF Model with New Test Data
% This script tests the trained APF model using strict_apf_ally_turn_1.mat

clc
clear
close all

%% Load the trained model
fprintf('Loading trained model...\n');
load('Sliced_strict_apf_net1.mat', 'multiSliceNet_mw9');

%% Load test data
fprintf('Loading test data...\n');
load('strict_apf_ally_turn_1.mat', 'apf_pq_r4');

% Transpose the elements in the dataset from 3xn to nx3
test_data = cell(size(apf_pq_r4, 1), 2);
for i = 1:size(apf_pq_r4, 1)
    for j = 1:size(apf_pq_r4, 2)
        % Transpose the 3xN matrix to Nx3
        apf_pq_r4{i, j} = apf_pq_r4{i, j}';
        test_data{i, j} = apf_pq_r4{i, j}(:, 1:4);
    end
end

%% Find trajectories with length > 100 (similar to training)
isLarge = cellfun(@(x) size(x, 1) > 100, test_data(:, 1));
longTrajectory = test_data(isLarge, :);

if isempty(longTrajectory)
    error('No trajectories with length > 100 found in test data');
end

fprintf('Found %d long trajectories in test data\n', size(longTrajectory, 1));

%% Randomly select a test trajectory
rng('shuffle'); % Initialize random seed
testIdx = randi(size(longTrajectory, 1));
fprintf('Selected trajectory index: %d\n', testIdx);

%% Set parameters
offset = 9; % Same as training

%% Extract test trajectory
testTrajectory = longTrajectory(testIdx, :);
allydata = testTrajectory{1};
enemydata = testTrajectory{2};

numTimeSteps = size(allydata, 1);
numPredictionTimeSteps = numTimeSteps - offset;

%% Prepare data normalization
% Collect all ally data for normalization
allAllyData = [];
for i = 1:size(longTrajectory, 1)
    allAllyData = [allAllyData; longTrajectory{i, 1}(:, 1:3)];
end

% Calculate normalization parameters from test data
muX_test = mean(allAllyData);
sigmaX_test = std(allAllyData, 0);

% Collect all enemy position data for output normalization
allEnemyData = [];
for i = 1:size(longTrajectory, 1)
    allEnemyData = [allEnemyData; longTrajectory{i, 2}(:, 1:2)];
end

muT_test = mean(allEnemyData);
sigmaT_test = std(allEnemyData, 0);

%% Prediction loop
fprintf('Running predictions...\n');
Y_pred = nan(numPredictionTimeSteps, 2);
Y_pred_norm = nan(numPredictionTimeSteps, 2);

% FILO filter parameters (same as training)
filo_length = 5;
Y_filter = nan(size(Y_pred));
buffer = zeros(filo_length, 2);
buffer_sum = [0, 0];

% Store test inputs and ground truth
XTrajTest = {};
TTrajTest = {};
j = 1;

for t = 1:numPredictionTimeSteps-1
    % Extract moving window
    Xmoving_window = allydata(t:t+offset, 1:3);
    Tmoving_window = enemydata(t+offset+1, 1:2);
    
    % Normalize input
    Xmoving_window_norm = (Xmoving_window - muX_test) ./ sigmaX_test;
    
    % Store for later analysis
    XTrajTest{j} = Xmoving_window;
    TTrajTest{j} = Tmoving_window;
    
    % Predict
    [Y_pred_norm(t, :), state] = predict(multiSliceNet_mw9, Xmoving_window_norm);
    multiSliceNet_mw9.resetState();
    
    % Denormalize prediction
    Y_pred(t, :) = Y_pred_norm(t, :) .* sigmaT_test + muT_test;
    
    % Apply FILO filter
    if t <= filo_length
        buffer(t, :) = Y_pred(t, :);
        buffer_sum = sum(buffer, 1);
        Y_filter(t, :) = buffer_sum / t;
    else
        buffer(1:filo_length-1, :) = buffer(2:filo_length, :);
        buffer(end, :) = Y_pred(t, :);
        buffer_sum = sum(buffer, 1);
        Y_filter(t, :) = buffer_sum / filo_length;
    end
    
    j = j + 1;
end

%% Generate visualization
fprintf('Generating visualization...\n');
figure('Position', [100, 100, 1000, 800]);

% Create color map for time steps
num_points = min(length(Y_pred), size(enemydata, 1) - offset - 1);
c = linspace(1, 10, num_points);

% Plot actual enemy positions
scatter(enemydata(offset+1:offset+num_points, 1), enemydata(offset+1:offset+num_points, 2), [], c, ...
    'DisplayName', 'Actual Enemy Positions');
hold on

% Plot starting points
scatter(enemydata(1, 1), enemydata(1, 2), 100, 'filled', 'd', ...
    'DisplayName', 'Enemy Start', 'MarkerFaceColor', [0.4940 0.1840 0.5560]);
scatter(allydata(1, 1), allydata(1, 2), 100, 'filled', 'd', ...
    'DisplayName', 'Ally Start', 'MarkerFaceColor', [0.8500 0.3250 0.0980]);

% Plot trajectories
plot(allydata(:, 1), allydata(:, 2), 'Color', [0 0.4470 0.7410], ...
    'LineWidth', 2, 'DisplayName', 'Ally Trajectory');
plot(enemydata(:, 1), enemydata(:, 2), '--', 'Color', [0.8 0.8 0.8], ...
    'LineWidth', 1.5, 'DisplayName', 'Enemy Trajectory (Ground Truth)');

% Plot predicted positions
scatter(Y_filter(1:num_points, 1), Y_filter(1:num_points, 2), [], c, 'filled', ...
    'DisplayName', 'Predicted Enemy Positions (Filtered)');

% Add raw predictions as smaller points
scatter(Y_pred(1:num_points, 1), Y_pred(1:num_points, 2), 20, c, '+', ...
    'DisplayName', 'Predicted Enemy Positions (Raw)');

% Formatting
xlabel('X Position', 'FontSize', 14);
ylabel('Y Position', 'FontSize', 14);
title(sprintf('APF Model Test - Trajectory %d', testIdx), 'FontSize', 16);
grid on;

% Add colorbar
colorbar;
colormap('parula');
cb = colorbar;
cb.Label.String = 'Prediction Step';
cb.Limits = [1, 10];
cb.Ticks = [1, 10];
cb.TickLabels = {'1', num2str(numPredictionTimeSteps-1)};

% Set axis limits based on data
x_min = min([allydata(:, 1); enemydata(:, 1); Y_filter(:, 1)]) - 2;
x_max = max([allydata(:, 1); enemydata(:, 1); Y_filter(:, 1)]) + 2;
y_min = min([allydata(:, 2); enemydata(:, 2); Y_filter(:, 2)]) - 2;
y_max = max([allydata(:, 2); enemydata(:, 2); Y_filter(:, 2)]) + 2;
xlim([x_min, x_max]);
ylim([y_min, y_max]);

legend('Location', 'best', 'FontSize', 10);
ax = gca;
ax.FontSize = 12;

hold off;

%% Calculate and display error metrics
% Calculate Euclidean distances
actual_positions = enemydata(offset+1:offset+num_points, 1:2);
predicted_positions = Y_filter(1:num_points, :);
distances = sqrt(sum((actual_positions - predicted_positions).^2, 2));

% Display metrics
fprintf('\n=== Test Results ===\n');
fprintf('Test Trajectory: %d\n', testIdx);
fprintf('Number of predictions: %d\n', length(distances));
fprintf('Mean prediction error: %.4f\n', mean(distances));
fprintf('Std prediction error: %.4f\n', std(distances));
fprintf('Min prediction error: %.4f\n', min(distances));
fprintf('Max prediction error: %.4f\n', max(distances));

%% Create error plot
figure('Position', [1150, 100, 600, 400]);
plot(distances, 'LineWidth', 2);
xlabel('Prediction Step', 'FontSize', 12);
ylabel('Euclidean Distance Error', 'FontSize', 12);
title('Prediction Error Over Time', 'FontSize', 14);
grid on;
ax = gca;
ax.FontSize = 11;

%% Save results
save('test_results.mat', 'testIdx', 'Y_pred', 'Y_filter', 'distances', ...
    'actual_positions', 'predicted_positions', 'XTrajTest', 'TTrajTest');
fprintf('\nResults saved to test_results.mat\n');