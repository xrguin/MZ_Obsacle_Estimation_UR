%% Test APF Model - Non-normalized Version
% This version uses the non-normalized model: Sliced_no_seg_apf_3layer.mat
% The model was trained without normalization, so this script:
% 1. Uses raw input data directly for prediction
% 2. Gets raw output data directly from the model
% Based on test_apf_model_v2.m structure using strict_apf_ally_turn_1.mat

clc
clear

%% Load the trained model
fprintf('Loading trained model (non-normalized)...\n');
load('Sliced_No_Seg_apf_3layer.mat', 'multiSliceNet');
multiSliceNet_mw9 = multiSliceNet;

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

%% Set parameters
offset = 9;

%% Initialize variables
Xmoving_window = zeros(offset, 3);
Tmoving_window = zeros(1, 2);
XTrajTest = {};
TTrajTest = {};

numLongTrajectory = size(longTrajectory, 1);
testIdx = 90;
testTrajectory = longTrajectory(testIdx, :);
allydata = testTrajectory{1};
enemydata = testTrajectory{2};
enemyInitial = enemydata(1, 1:3);

numTimeSteps = size(allydata, 1);
numPredictionTimeSteps = numTimeSteps - offset;
j = 1;
prediction_time_list = []; 

Y_reg = nan(numPredictionTimeSteps, 2); % Direct predictions without normalization

% FILO filter
% Define filter parameters
filo_length = 5;  % Length of the moving average window
Y_filter = nan(size(Y_reg));  % Preallocate filtered output
buffer = zeros(filo_length, 2);  % Circular buffer for moving average
buffer_sum = [0, 0];  % Sum of values in the buffer

theta_threshold = deg2rad(4);
theta = allydata(:,3);
delta_theta = mod(diff(theta) + pi, 2*pi) - pi;

predictionStarted = false;
predictionStartStep = -1;

current_theta = allydata(3,3);
current_theta_enemy = enemydata(3,3);

%% Prediction loop
for t = 1:numPredictionTimeSteps-1
    % If all ally inputs have a nonzero angular velocity,
    % then store as an input data

   % if  ~predictionStarted && (abs(current_theta - allydata(t,3) )> theta_threshold)
     %       || abs(current_theta_enemy - enemydata(t,3))>theta_threshold)
    %    predictionStarted = true;
   %     predictionStartStep = t;
   %     predictionthetastart = allydata(t,3);
   %     fprintf("Start predicting at time step %d \n", ...
   %         t+offset);
   % end

  %  if predictionStarted
      Xmoving_window = allydata(t:t+offset, 1:3);
      Tmoving_window = enemydata(t+offset+1, 1:2);
      
      % No normalization needed - use raw data directly
      XTrajTest{j} = Xmoving_window;
      TTrajTest{j} = Tmoving_window;
      prediction_time_list(end+1) = t;
      
      % Predict with raw input (no normalization)
      Y_reg(t, :) = predict(multiSliceNet_mw9, XTrajTest{j});
      
      j = j + 1;
    
      if t <= filo_length
          buffer(t, :) = Y_reg(t, :);
          buffer_sum = sum(buffer, 1);
          Y_filter(t, :) = buffer_sum / t;  % For the first few points
      else
         % if t > predictionStartStep + 4
           % if t > predictionStartStep + 5
              buffer(1:filo_length-1, :) = buffer(2:filo_length, :);
              buffer(end, :) = Y_reg(t, :);
              buffer_sum = sum(buffer, 1);
              Y_filter(t, :) = buffer_sum / filo_length;  % Full window average
         %   else
          %    buffer(1:filo_length,:) = Y_reg(t-filo_length+1:t,:);
         %     buffer_sum = sum(buffer, 1);
          %    Y_filter(t, :) = buffer_sum / filo_length;  % Full window average
          %  end
          %else
          %  Y_filter(t,:) = Y_reg(t,:);
         % end
      end
   % end
end

%% Find sections with angular velocity
numWindows = size(allydata, 1) - offset;
validWindows = false(numWindows, 1);

angular_velocity = allydata(:, 4);
non_zero_dphi = abs(angular_velocity) > 0.001;
section_start = find(diff([0; non_zero_dphi; 0]) == 1);
section_end = find(diff([0; non_zero_dphi; 0]) == -1);

plot_idx = (section_start + offset) <= length(non_zero_dphi);
section_start = section_start(plot_idx);
section_end = section_end(plot_idx);

if ~isempty(section_end) && section_end(end) > length(non_zero_dphi)
    section_end(end) = length(non_zero_dphi);
end

% Find the coordinates based on the index
if ~isempty(section_start)
    start_x = allydata(section_start, 1);
    start_y = allydata(section_start, 2);
    end_x = allydata(section_end, 1);
    end_y = allydata(section_end, 2);
end

%% 正确地找到误差大的 XTrajTest 索引
gt_pos = enemydata(prediction_time_list + offset + 1, 1:2);
pred_pos = Y_reg(prediction_time_list, :);
error_vec = vecnorm(pred_pos - gt_pos, 2, 2);

error_threshold = 1.5;
bad_mask = error_vec > error_threshold;
bad_j_idx = find(bad_mask);  % 对应 XTrajTest 中的索引 j

fprintf('Found %d predictions with error > %.2f m\n', length(bad_j_idx), error_threshold);

% 可视化这些 X 窗口
figure;
hold on
title(sprintf('Input Windows for Bad Predictions (error > %.2f m)', error_threshold))
xlabel('X'); ylabel('Y'); axis equal; grid on

for i = 1:length(bad_j_idx)
    j_idx = bad_j_idx(i);
    xwindow = XTrajTest{j_idx};  % Already raw data, no denormalization needed
    plot(xwindow(:,1), xwindow(:,2), '-o', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Step %d', prediction_time_list(j_idx)+offset));
end
legend('show');

%% Plotting
figure;
c = linspace(1, 10, length(Y_reg));
scatter(enemydata(offset+1:end, 1), enemydata(offset+1:end, 2), [], c, ...
    'DisplayName', 'Actual Enemy Locations');
hold on

scatter(enemydata(1, 1), enemydata(1, 2), 60, 'filled', 'd', ...
    'DisplayName', 'Enemy Start Point', 'MarkerFaceColor', [0.4940 0.1840 0.5560])
plot(allydata(:, 1), allydata(:, 2), 'Color', [0 0.4470 0.7410], 'LineWidth', 2, ...
    'DisplayName', 'Ally Trajectory')
plot(enemydata(:, 1), enemydata(:, 2), '--', 'DisplayName', 'Enemy Trajectory')

scatter(allydata(1, 1), allydata(1, 2), 60, 'filled', 'd', 'DisplayName', 'Ally Start Point', ...
    'MarkerFaceColor', [0.8500 0.3250 0.0980])
scatter(Y_filter(:, 1), Y_filter(:, 2), [], c, 'filled', ...
    'DisplayName', 'Predicted Enemy Locations')

xlabel("X")
ylabel('Y')
xlim([-5, 15])
ylim([-5, 15])
colorbar;
colormap("parula");
cb = colorbar;
cb.Label.String = 'Predict Steps';
cb.Limits = [1, 10];
cb.Ticks = [1 10];  % Place ticks at the ends of the color bar
cb.TickLabels = {'1', num2str(numPredictionTimeSteps)};  % Set the labels at these ticks to '1' and the actual number
legend('show')
ax = gca;
ax.FontSize = 16;
hold off

%% Display results
fprintf('\n=== Test Results (Non-normalized Model) ===\n');
fprintf('Test Trajectory: %d\n', testIdx);
fprintf('Number of predictions: %d\n', length(Y_reg));
fprintf('Trajectory length: %d time steps\n', numTimeSteps);
fprintf('Prediction window: %d time steps\n', numPredictionTimeSteps);

% Calculate error statistics
all_errors = vecnorm(Y_reg - enemydata(offset+1:end, 1:2), 2, 2);
valid_errors = all_errors(~isnan(all_errors));

if ~isempty(valid_errors)
    fprintf('\n=== Error Statistics ===\n');
    fprintf('Mean error: %.4f m\n', mean(valid_errors));
    fprintf('Std error: %.4f m\n', std(valid_errors));
    fprintf('Min error: %.4f m\n', min(valid_errors));
    fprintf('Max error: %.4f m\n', max(valid_errors));
    fprintf('Median error: %.4f m\n', median(valid_errors));
end