clear;clc;

offset = 9;

load('Sliced_strict_apf_net1.mat', 'multiSliceNet_mw9');
load('strict_apf_ally_turn_1.mat','apf_pq_r4');

Xmoving_window = zeros(offset,3);
Tmoving_window = zeros(1,2);
XTrajTest = {};
TTrajTest = {};

numLongTrajectory = size(apf_pq_r4,1);
testIdx = 2;
testTrajectory = apf_pq_r4(testIdx,:);
allydata = testTrajectory{1};
enemydata = testTrajectory{2};
enemyInitial = enemydata(1,1:3);

numTimeSteps = size(allydata,1);
numPredictionTimeSteps = numTimeSteps - offset;
j = 1;

Y_norm= nan(numPredictionTimeSteps,2); % Out of plotting range
Y_reg = Y_norm;

% FILO filter
% Define filter parameters
filo_length = 5;  % Length of the moving average window
Y_filter = nan(size(Y_reg));  % Preallocate filtered output
buffer = zeros(filo_length,2);  % Circular buffer for moving average
buffer_sum = [0,0];  % Sum of values in the buffer



for t = 1:numPredictionTimeSteps-1
    % If all ally inputs have a nonzero angular velocity,
    % then store as an input data
    Xmoving_window = allydata(t:t+offset,1:3);
    Tmoving_window = enemydata(t+offset+1,1:2);
    XTrajTest{j} = Xmoving_window;
    TTrajTest{j} = Tmoving_window;
    [Y_reg(t,:),state] = predict(multiSliceNet_mw9,XTrajTest{j});
    multiSliceNet_mw9.resetState();
    j = j + 1;
    

    if t <= filo_length
        buffer(t,:) = Y_reg(t,:);
        buffer_sum = sum(buffer,1);
        Y_filter(t,:) = buffer_sum / t;  % For the first few points
    else
        buffer(1:filo_length-1,:) = buffer(2:filo_length,:);
        buffer(end,:) = Y_reg(t,:);
        buffer_sum = sum(buffer,1);
        Y_filter(t,:) = buffer_sum / filo_length;  % Full window average
        
    end
    
end
% Y_reg = Y_norm .* sigmaT_train + muT_train;
%%
numWindows = size(allydata, 1) - offset;
validWindows = false(numWindows, 1);

angular_velocity = allydata(:,4);
non_zero_dphi = abs(angular_velocity) > 0.001;
section_start = find(diff([0;non_zero_dphi;0]) == 1);
section_end = find(diff([0;non_zero_dphi;0]) == -1);

plot_idx = (section_start + offset ) <= length(non_zero_dphi);
section_start = section_start(plot_idx);
section_end = section_end(plot_idx);

if section_end(end) > length(non_zero_dphi)
    section_end(end) = length(non_zero_dphi);
end


% Find the coordinates based on the index

start_x = allydata(section_start,1);
start_y = allydata(section_start,2);
end_x = allydata(section_end,1);
end_y = allydata(section_end,2);
%%
figure;
c = linspace(1,10, length(Y_reg));
scatter(enemydata(offset+1:end,1), enemydata(offset+1:end,2),[],c, ...
    'DisplayName','Actual Enemy Locations');
hold on
% 
% scatter(enemydata(offset+section_start,1),enemydata(offset+section_start,2),80,'magenta','filled','o', ...
%     'DisplayName','Non-straight Starting Point')
% plot(allydata(1:offset,1), allydata(1:offset,2),'b','LineWidth',2,'DisplayName','Ally')
scatter(enemydata(1,1),enemydata(1,2),60,'filled','d', ...
    'DisplayName','Enemy Start Point','MarkerFaceColor',[0.4940 0.1840 0.5560])
plot(allydata(:,1),allydata(:,2),'Color',[0 0.4470 0.7410],'LineWidth',2,'DisplayName','Ally Trajectory')
plot(enemydata(:,1),enemydata(:,2),'--','DisplayName','Enemy Trajectory')
% plot(allydata(1:offset,1), allydata(1:offset,2),'g','LineWidth',2, ...
%     'DisplayName','First Input (Ally)')
scatter(allydata(1,1),allydata(1,2),60,'filled','d','DisplayName','Ally Start Point','MarkerFaceColor',[0.8500 0.3250 0.0980])
scatter(Y_filter(:,1), Y_filter(:,2),[],c,'filled','DisplayName','Predicted Enemy Locations')
% first_not_straight = XTrajTest{section_start(1)};
% plot(first_not_straight(2:end,1), first_not_straight(2:end,2),'g','LineWidth',2, ...
%     'DisplayName','First Non-straint Input Data')
% scatter(Y_reg(section_start,1),Y_reg(section_start,2),80,'red','filled','o', ...
%     'DisplayName','Non-straight Starting Point')
xlabel("X")
ylabel('Y')
xlim([-5,15])
ylim([-5,15])
colorbar;
colormap("parula");
cb = colorbar;
cb.Label.String = 'Predict Steps';
cb.Limits = [1,10];
cb.Ticks = [1 10];  % Place ticks at the ends of the color bar
cb.TickLabels = {'1', numPredictionTimeSteps};  % Set the labels at these ticks to '1' and '92'
legend('show')
ax = gca;
ax.FontSize = 16;
hold off